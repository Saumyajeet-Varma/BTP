# Endsem-imp-pipeline-02.py — Documentation

This document describes **`Endsem_prep/Endsem-imp-pipeline-02.py`**: the same **two-stage** Car-Hacking pipeline as **`Endsem-imp-pipeline-01.py`**, extended with **synthetic probe signals** so you can **sanity-check Stage 2** (decoder-based and optional uniform fakes) without claiming they are real CAN logs.

For shared concepts (data loading, Stage 1 backbone, AAE training, calibration threshold), see **`Endsem-imp-pipeline-01.md`**; this file focuses on **what is new or different** in pipeline-02.

---

## 1. Purpose

**File:** `Endsem_prep/Endsem-imp-pipeline-02.py`

**Pipeline (same as 01):**

1. **Stage 1** — Multi-class classifier (CANForge-style + focal loss): **Normal**, **DoS**, **Fuzzy**, **Gear**, **RPM**.
2. **Stage 2** — AAE trained **only on true Normal** rows from the training split; combined **reconstruction + latent** score and a **percentile threshold** on calibration normals that pass Stage 1.
3. **Hybrid rule** — If Stage 1 predicts an attack class, that label wins. If Stage 1 predicts **Normal**, Stage 2 scores the sample; if **score > threshold**, output **`zero_day`** (unknown / suspicious normal-looking traffic).

**What pipeline-02 adds:**

- **`FakeSignalGenerator`** — Uses the trained **`Decoder`** only: sample **`z ~ N(0, σ² I)`**, output **`x = Decoder(z)`** in **`[0,1]^d`** (same dimension as scaled features). This is a **QA proxy** for “odd” vectors in feature space, **not** synthetic CAN frames from the log format.
- **`uniform_fake_features`** — Optional i.i.d. **Uniform(0,1)** vectors of length **`d`**, scored the same way for comparison.
- **Printed QA** — Mean anomaly scores and **fraction flagged** for decoder fakes, uniform fakes (if enabled), and calibration normals; plus a **balanced** binary task (**real_normal_cal** vs **decoder_fake**) with accuracy / report / confusion matrix.
- **Extra plot** — Histogram of **cal normals** vs **decoder fakes** vs **uniform fakes** with the **threshold** line.

**Style:** Same as pipeline-01 — section banners, optional Colab mount, linear execution, matplotlib/seaborn at the end.

---

## 2. Execution Flow (Top to Bottom)

```
Optional Colab mount
    ↓
Imports, seeds, hyperparameters (including fake-probe tunables), device
    ↓
Classes: FocalLoss, SEBlock, CANForgeStage1, Encoder, Decoder,
          LatentDiscriminator, FakeSignalGenerator
    ↓
Helpers: uniform_fake_features, parse_line, load_full_dataframe, …
    ↓
if data_path missing → print error and stop
else → Stage 1 → Stage 2 → threshold → synthetic QA block → test hybrid metrics → plots
```

---

## 3. Configuration Blocks

### 3.1 Colab vs local

Same as pipeline-01: tries **`google.colab.drive`**, otherwise **`data_path = r"9) Car-Hacking Dataset"`**.

### 3.2 Reproducibility

**`RANDOM_STATE = 42`** for NumPy and PyTorch; **`uniform_fake_features`** uses the same seed when no RNG is passed.

### 3.3 Tunables — shared with pipeline-01

| Name | Role |
|------|------|
| `USE_SUBSET`, `MAX_NORMAL`, `MAX_PER_ATTACK_FILE` | Dataset caps |
| `BATCH_SIZE`, `STAGE1_*`, `STAGE1_VAL_FRAC` | Stage 1 |
| `STAGE2_NORMAL_PERCENTILE` | Percentile → Stage 2 **threshold** on cal normals (Stage 1 predicted Normal) |
| `AAE_*`, `LATENT_DIM`, `LATENT_WEIGHT` | Stage 2 AAE |

### 3.4 Tunables — synthetic probes only (pipeline-02)

| Name | Default (script) | Role |
|------|------------------|------|
| `NUM_FAKE_SAMPLES` | `8000` | How many decoder fakes (and uniform fakes if enabled) to score |
| `FAKE_LATENT_STD` | `2.5` | Standard deviation per latent dimension for **`z`** before **`Decoder(z)`** |
| `INCLUDE_UNIFORM_FAKE` | `True` | Also score **`Uniform(0,1)^d`** vectors |

**Interpretation:** Larger **`FAKE_LATENT_STD`** pushes **`z`** farther from the origin; decoder outputs remain in **`[0,1]^d`** due to **Sigmoid**. Tuning affects how “extreme” the decoder probes are, not real attack semantics.

---

## 4. New Components

### 4.1 `FakeSignalGenerator`

- Holds a reference to **`decoder`**, **`latent_dim`**, **`latent_std`**.
- **`sample(n, device)`** / **`sample_numpy(n, device)`**: **`torch.no_grad()`**, decoder in **eval**, **`z = randn * latent_std`**, return **`decoder(z)`**.

### 4.2 `uniform_fake_features(n, n_features, rng=None)`

- Returns **`float32`** array **`(n, n_features)`** with entries **`Uniform(0, 1)`**.
- Independent of the trained model; useful as a **simple synthetic negative** baseline in score space.

---

## 5. Threshold and Synthetic QA (after `threshold` is fixed)

**Calibration scores (`val_scores`)** — Same definition as pipeline-01: rows in **`X_cal`** that are **true Normal** and **predicted Normal** by Stage 1 (with fallback if too few samples).

Then:

1. Build **`fake_decoder_np`** with **`FakeSignalGenerator`**.
2. Optionally build **`fake_uniform_np`** and **`fake_scores_uni`**.
3. **`combined_scores`** uses the **same** **`recon_mean/std`**, **`lat_mean/std`**, **`LATENT_WEIGHT`** as for real data.

**Printed block:**

- Decoder fakes: mean score, **% with score > threshold**.
- Uniform fakes (if enabled): same.
- Cal normals: mean score, **% with score ≤ threshold** (“pass”).

**Balanced QA:** Draw **`n_bal = min(len(val_scores), NUM_FAKE_SAMPLES)`** calibration scores (random subset without replacement) and pair with the **first `n_bal`** decoder fake scores. Labels: **0** = real_normal_cal, **1** = decoder_fake; prediction **1** if **score > threshold**. Reports accuracy, precision, recall, F1, **`classification_report`**, **`confusion_matrix`**.

This answers: “If we treat decoder fakes as positives, does the threshold separate them from held-out calibration normals?” — **not** “does this detect real zero-days?”

---

## 6. Test-Time Hybrid Labels (pipeline-02)

For each test index **`i`**:

1. If **`y_test_pred[i] != normal_idx`** → final label = **`le.inverse_transform`** class name (**DoS**, **Fuzzy**, **Gear**, **RPM**, **Normal** as applicable).
2. Else if **`test_scores[i] > threshold`** → **`zero_day`**.
3. Else → **`Normal`**.

Attack names come from **`le.inverse_transform`** (same pattern as pipeline-01).

---

## 7. Metrics Printed (vs pipeline-01)

Pipeline-02 includes:

- Stage 1 full report and confusion matrix on the **test** set.
- Synthetic probe summary + **balanced QA** (real_normal_cal vs decoder_fake).
- Stage 2 **gated** metrics on test rows where Stage 1 predicted **Normal** (true_normal vs true_attack, flag vs pass).
- **Hybrid** multi-class style report: **`true_names`** vs **`final_label`** (union of labels → confusion matrix may include **`zero_day`**).
- Hybrid **binary** attack detection and **normal clearance** / **Stage 2 FP on normals**.

Pipeline-02 **does not** print optional **ROC AUC** on raw test scores (present in some versions of pipeline-01).

---

## 8. Visualizations

Same first four ideas as pipeline-01 where applicable:

1. Stage 1 train vs val loss.
2. Stage 1 test confusion heatmap.
3. Stage 2 mean AE reconstruction loss per epoch.
4. Stage 2 score histograms on **test**: Normal vs Attack + threshold.

**Additional (pipeline-02):**

5. **Calibration normals vs decoder fakes vs uniform fakes** (three overlays) + threshold — qualitative separation check.

Then (if **`n_s2 > 0`**): Stage 2 gated confusion heatmap. Finally: **hybrid** confusion heatmap (dynamic size with label count).

---

## 9. Comparison to `Endsem-imp-pipeline-01.py`

| Aspect | Pipeline-01 | Pipeline-02 |
|--------|-------------|-------------|
| Core stages | Stage 1 + AAE + hybrid rule | Same |
| Calibration path | True Normal ∧ Stage 1 Normal on **`X_cal`** | Same |
| Hybrid attack labels | **`inverse_transform`** names + **`zero_day`** | Same |
| Synthetic data | None | Decoder **`z→x`** fakes + optional uniform **`[0,1]^d`** |
| Extra metrics | — | Probe means, % flagged, balanced QA matrix |
| Extra plot | — | Cal vs decoder vs uniform histogram |
| ROC AUC on scores | May appear | Not included |

---

## 10. How to Run

```bash
cd Endsem_prep   # or project root where data_path resolves
python Endsem-imp-pipeline-02.py
```

Place the **Car-Hacking** dataset where **`data_path`** points (adjust the **`data_path`** assignment for local layout).

---

## 11. File Reference (Quick Map)

| Lines (approx) | Content |
|----------------|---------|
| 1–71 | Header, Colab, imports, seeds, **`data_path`**, tunables (**including fake probes**), `device` |
| 74–203 | Loss, CNN/LSTM Stage 1, **Encoder / Decoder / Disc**, **`FakeSignalGenerator`**, **`uniform_fake_features`** |
| 206–482 | Parsing, **`load_full_dataframe`**, **`predict_*`**, **`combined_scores`**, **`train_stage1_classifier`**, **`train_stage2_aae`** |
| 485–789 | Main: load → split → Stage 1 → Stage 2 → threshold → **synthetic QA** → hybrid metrics → **six (+ conditional) plots** |

---

*Companion documentation for `Endsem_prep/Endsem-imp-pipeline-02.py`; pair with `Endsem-imp-pipeline-01.md` for the full two-stage baseline description.*
