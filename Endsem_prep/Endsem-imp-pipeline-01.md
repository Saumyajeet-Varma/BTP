# Endsem-imp-pipeline-01.py — Documentation

This document explains the **two-stage** CAN intrusion script: how it is organized (style), what each block does, and how it differs from running the notebook end-to-end in one monolithic flow.

---

## 1. Purpose

**File:** `Endsem_prep/Endsem-imp-pipeline-01.py`

**Pipeline:**

1. **Stage 1** — Supervised **multi-class** classifier (CANForge-style backbone): assigns **Normal** or one of the **known** attack labels (DoS, Fuzzy, Gear, RPM).
2. **Stage 2** — **AAE** (denoising autoencoder + latent discriminator) trained **only on true Normal** samples. It scores **reconstruction + latent** abnormality for **second-pass** screening.
3. **Deployment rule** — Only samples that Stage 1 calls **Normal** are scored by Stage 2. The **threshold** is set from **calibration data** that reflects this path: rows that are **truly Normal** *and* **predicted Normal** by Stage 1.

**Goal:** Known attacks are handled in Stage 1; traffic that “looks Normal” to Stage 1 is screened for **zero-day / unknown** behavior in Stage 2, while trying to **minimize false alarms** on legitimate normal traffic that passes Stage 1.

**Style:** Section banners (`# ===...===`), optional **Google Colab** mount, linear **script execution** (no `main()`), and **matplotlib/seaborn** plots at the end — aligned with `codes/model2025_stage2_cursor.py`.

---

## 2. Execution Flow (Top to Bottom)

```
Optional Colab mount
    ↓
Imports, seeds, hyperparameters, device
    ↓
Class definitions (FocalLoss, SEBlock, CANForgeStage1, Encoder, Decoder, LatentDiscriminator)
    ↓
Helper functions (parse_line, paths, load_full_dataframe, train_*, predict_*, combined_scores)
    ↓
if data_path missing → print error and stop
else → full pipeline + plots
```

---

## 3. Configuration Blocks

### 3.1 Colab vs local

- Tries `from google.colab import drive` and `drive.mount(...)`.
- If import fails, **`_IN_COLAB = False`** and **`data_path`** defaults to `9) Car-Hacking Dataset`.
- If Colab succeeds, **`data_path`** points under Drive (edit to match your layout).

### 3.2 Reproducibility

- **`RANDOM_STATE = 42`** applied to **NumPy** and **PyTorch** manual seed.

### 3.3 Tunables (important)

| Name | Role |
|------|------|
| `USE_SUBSET`, `MAX_NORMAL`, `MAX_PER_ATTACK_FILE` | Limit dataset size |
| `BATCH_SIZE` | Stage 1 loader |
| `STAGE1_EPOCHS`, `STAGE1_PATIENCE` | Classifier training |
| `STAGE1_VAL_FRAC` | Fraction of the **80% train pool** held out for **calibration** (Stage 1 val + Stage 2 threshold stats) |
| `STAGE2_NORMAL_PERCENTILE` | Percentile on **cal** combined scores (true Normal ∧ Stage1 Normal) → **threshold** |
| `AAE_EPOCHS`, `AAE_BATCH`, `DISC_STEPS`, `INPUT_NOISE_STD`, `LATENT_DIM`, `LATENT_WEIGHT` | Stage 2 AAE behavior |

---

## 4. Stage 1 — Models and Loss

### 4.1 `FocalLoss`

- Wraps **`cross_entropy`** with optional **class weights**, **label smoothing** (default 0.05), and **focal** modulation `(1 - pt)^gamma * CE` to down-weight easy examples.
- Helps with **imbalanced** CAN classes compared to plain CE alone.

### 4.2 `SEBlock`

- Same idea as the notebook: channel attention after global average over the **sequence** dimension.

### 4.3 `CANForgeStage1`

- **Multi-scale** `Conv1d` (kernels 1, 3, 5) → 96 channels.
- **SE**, dropout, **residual** conv block.
- **Two BiLSTM** layers with batchnorm-on-time trick, **residual** between LSTM outputs.
- **Global mean** over sequence → MLP → **logits** (`num_classes`).

Input tensor shape: **`(batch, 1, 16)`** (one channel, sixteen “time” steps).

### 4.4 `train_stage1_classifier`

- **AdamW** + **ReduceLROnPlateau** on **validation** focal loss.
- Each epoch: train pass (accumulate mean train loss), eval pass on **cal** loader for val loss.
- **Early stopping** when val loss does not improve for **`STAGE1_PATIENCE`** epochs.
- Restores best weights; returns **elapsed seconds** and **loss curves** for plotting.

---

## 5. Data Pipeline

### 5.1 `parse_line` / `normal_txt_path`

- Same regex idea as the notebook for the **normal** text log.
- **`normal_txt_path`** checks **`normal_run_data.txt`** at dataset root **or** **`normal_run_data/normal_run_data.txt`** (Colab-style layout).

### 5.2 `load_full_dataframe`

- Reads capped normal lines, loads four attack CSVs, applies **Flag**-based labeling (`R` / `T`).
- Builds the **same 16 features** as CANForge: base 10 + `IAT`, `CAN_ID_freq`, `byte_entropy`, `byte_sum`, `byte_range`, `byte_std`.
- Returns **`full_df`** and the **`features`** list.

### 5.3 Splits and scaling

1. **Stratified** 80/20 → **`X_trf_s`**, **`X_test_s`** (scaler fit on train portion of 80%).
2. From **`X_trf_s`**, another **stratified** split by **`STAGE1_VAL_FRAC`** → **`X_train`** (fits Stage 1 + supplies normals for AAE) and **`X_cal`** (Stage 1 validation + **threshold calibration**).

Tensors for Stage 1: **`unsqueeze(1)`** → `(N, 1, 16)`.

---

## 6. Stage 2 — AAE

### 6.1 Modules

- **`Encoder` / `Decoder`**: MLP autoencoder; decoder ends with **Sigmoid** (matches MinMax).
- **`LatentDiscriminator`**: spectral-normalized MLP; **BCEWithLogits** adversarial game on latents.

### 6.2 Training subset

- **`X_normal_train`** = rows in **`X_train`** whose label is **Normal**.
- Further split → **`Xn_tr`** (AAE training), **`Xn_val`** (tensor on device for val MSE each epoch).

### 6.3 `train_stage2_aae`

Each epoch:

1. **Denoising recon**: noisy input → encode → decode; **SmoothL1** vs clean; step **encoder+decoder**.
2. **Discriminator** `DISC_STEPS` times: Gaussian `z` vs `enc(real)` detached.
3. **Encoder adversarial** step: fool discriminator on `enc(real)`.

Tracks **mean AE loss per epoch** in **`ae_loss_log`** for the green curve plot. Early stop on **validation MSE** patience; reload best **enc/dec/disc**.

---

## 7. Threshold (Stage 2, calibration path)

On **`X_cal`**:

- **`cal_normal_mask`**: true label is Normal.
- **`passed_s1`**: Stage 1 prediction is Normal.
- **`pipeline_normal_mask`**: **both** (this is the operational path for benign traffic).

If too few samples (&lt; 50), falls back to **all true cal normals** (with a warning).

From those rows:

- Compute **reconstruction MSE** and **`||z||`**, then **mean/std** for z-scoring.
- **`combined_scores`** = weighted sum of **absolute z-scores** (reconstruction vs latent), using **`LATENT_WEIGHT`**.
- **`threshold`** = **`np.percentile(val_scores, STAGE2_NORMAL_PERCENTILE)`** on that calibration set.

Higher percentile → stricter “must look like calibration normals” → fewer normals slip through as final Normal, but more risk of flagging benign edge cases.

---

## 8. Test-Time Decisions

For each test index `i`:

1. If **`y_test_pred[i] != normal_idx`** → **`Known:<name>`** (Stage 1 wins).
2. Else if **`test_scores[i] > threshold`** → **`ZeroDay`**.
3. Else → **`Normal`**.

**Metrics printed:**

- Stage 1 accuracy / weighted F1 and full **`classification_report`**.
- Hybrid **binary** (any attack vs not): accuracy, precision, recall, F1.
- Optional **ROC AUC** on raw **`test_scores`** vs true attack/normal (not stage-gated).
- **Normal clearance rate**: fraction of **true normals** with final label **`Normal`**.
- **Stage 2 FPR on normals**: fraction of **true normals** not ending as **`Normal`** (hybrid false alarm on benign).

---

## 9. Visualizations (End of Script)

1. **Stage 1** — train vs validation focal loss per epoch.
2. **Stage 1** — seaborn heatmap of **confusion matrix** on the test set.
3. **Stage 2** — mean AE reconstruction loss per epoch.
4. **Stage 2** — overlapping histograms of **combined scores** for true Normal vs Attack on the **full** test set, with **vertical threshold** line.

---

## 10. Comparison to `CANForge_PyTorch_GPU_v2.ipynb`

| Aspect | Notebook | This script |
|--------|----------|-------------|
| Stage 1 loss | Weighted **CrossEntropyLoss** | **FocalLoss** + label smoothing + class weights |
| Stage 1 optimizer | Adam | **AdamW** + grad clip |
| Calibration | AAE threshold from notebook’s val normal scores | Threshold from **true Normal ∧ Stage1 Normal** on **`X_cal`** |
| Hybrid | Same high-level rule | Same rule, explicit **Known:** prefix |
| Structure | Jupyter cells | Single `.py` with banners + `if data_path` guard |
| Plots | Mixed across cells | Four figures at end |

The **feature set** and **CANForge-style** backbone intent match the notebook; the script is tuned for a **clean two-stage story** and **stricter normal calibration path**.

---

## 11. How to Run

```bash
cd Endsem_prep   # or project root where data_path resolves
python Endsem-imp-pipeline-01.py
```

Ensure the **Car-Hacking** dataset is visible at **`data_path`**. For local runs, set **`data_path`** in the **Dataset Path** section if your folder name differs.

---

## 12. File Reference (Quick Map)

| Lines (approx) | Content |
|----------------|---------|
| 1–77 | Headers, Colab, imports, seeds, paths, tunables, `device` |
| 79–209 | PyTorch classes |
| 212–320 | Parsing + `load_full_dataframe` |
| 323–471 | `predict_stage1_labels`, `combined_scores`, `train_stage1_classifier`, `train_stage2_aae` |
| 474+ | Main pipeline: load → split → Stage 1 → Stage 2 → threshold → decisions → prints → plots |

---

*Generated as companion documentation for `Endsem_prep/Endsem-imp-pipeline-01.py`.*
