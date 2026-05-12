# endsem_new_pipeline.py — Documentation

Companion to **[`Endsem_prep/endsem_new_pipeline.py`](endsem_new_pipeline.py)**. For the **CANGuard** paper in prose form, see **[`CANGuard_base_paper_summary.md`](CANGuard_base_paper_summary.md)**.

---

## 1. Purpose

End-to-end **two-stage** CAN intrusion detection that:

1. Uses a **CANGuard-inspired** **Stage 1** (Conv1D stack → stacked **BiGRU** → **additive attention** → MLP) on **sliding temporal windows** of engineered features.
2. Uses a **denoising adversarial autoencoder (AAE)** **Stage 2** trained **only on ground-truth Normal** windows (flattened), scoring traffic that **Stage 1 classifies as Normal**.
3. Produces **hybrid** labels: known attack name if Stage 1 is non-Normal; **`zero_day`** if Stage 1 is Normal **and** anomaly score **> threshold**; else **Normal**.

**Dataset:** **Car-Hacking** (same 16-D per-message feature construction as `Endsem-imp-pipeline-01`), sorted by timestamp. **Note:** Car-Hacking has **no true unknown class**; `zero_day` is an **operational flag** on the Stage 2 path (see §5).

---

## 2. Data flow

```
Car-Hacking files → parse / CSV → 16 features + Label → MinMax [0,1] on train
    → stratified train | cal | test (row-level)
    → sliding windows (SEQ_LEN consecutive rows) per split
        X_window: (N, F, SEQ_LEN), y_window: label of last row in window
```

Windows are built **inside each split** so a window does not span train into test.

---

## 3. Stage 1 — CANGuard-style

| Item | Implementation |
|------|----------------|
| Input shape | `(batch, n_features, seq_len)` — multivariate series along time |
| CNN | Three **Conv1d** stages (64 → 128 → 256 filters, kernel 3), **BN**, **ReLU**, **MaxPool1d(2)**, **dropout 0.3** |
| Temporal | Two **bidirectional GRU** layers (hidden 128 then 64) |
| Attention | **tanh** projection + **softmax** weights over time → context vector |
| Head | **Linear** 256 → **ReLU** → **dropout** → 128 → **ReLU** → **dropout** → logits |
| Loss | **Focal loss** + label smoothing + **class weights** (balanced) |
| Optimizer | **Adam**, lr `STAGE1_LR`, weight decay `STAGE1_WEIGHT_DECAY` (L2-style), **grad clip 1.0**, **ReduceLROnPlateau** |
| Early stop | `STAGE1_PATIENCE` on validation loss |

This follows the **paper’s decomposition** (spatial CNN + temporal GRU + attention) while adapting to **Car-Hacking** labels: **Normal**, **DoS**, **Fuzzy**, **Gear**, **RPM**.

---

## 4. Stage 2 — AAE on normals

| Item | Detail |
|------|--------|
| Input | **Flattened** window: dimension **`n_features * SEQ_LEN`**, values in **[0, 1]** after MinMax |
| Training set | **Only** windows whose **true** label is **Normal** (from the **train** split, after windowing) |
| Model | **Encoder** → **Decoder** (sigmoid output); **latent discriminator** adversarial term on `z` |
| Threshold | **`STAGE2_NORMAL_PERCENTILE`** of anomaly scores on **calibration** windows that are **true Normal ∧ Stage 1 predicted Normal** |
| Score | Z-normalized **reconstruction MSE** and **‖z‖** vs reference stats (`STAGE2_SCORE_STATS`: **`train_normal`** or **`cal_normal`**); aggregate via **`max`** or **`blend`** (`STAGE2_SCORE_AGG`, `LATENT_WEIGHT` for blend) |

---

## 5. Printed metrics (test set)

The script prints **`accuracy_score`**, **`precision`**, **`recall`**, **`f1_score`**, and **`confusion_matrix`** for:

| Block | What it measures |
|-------|------------------|
| **Stage 1** | Multiclass window labels vs Stage 1 predictions — **weighted** and **macro** precision / recall / F1, full **`confusion_matrix`**, then `classification_report`. |
| **Stage 2 (gated)** | Binary: only samples where **Stage 1 = Normal**; true 0=normal / 1=attack vs pred from **score > threshold** — accuracy, precision, recall, F1, **2×2 confusion_matrix**. |
| **Stage 2 (all-score)** | Diagnostic binary on **all** test windows: ground-truth attack vs **score > T** (ignores Stage 1 gate) — same scalar + **2×2** matrix. |
| **Combined (multiclass)** | True string label vs **hybrid** final string (DoS, …, Normal, **zero_day**) — weighted/macro P/R/F1 and full **confusion_matrix**. |
| **Combined (binary)** | True “any attack” vs hybrid “not final Normal” — binary P/R/F1 and **2×2** matrix. |
| **METRICS SUMMARY** | One-line recap of Stage 1, both Stage 2 views, and both hybrid views. |

Plots still include heatmaps for **Stage 1** multiclass and **hybrid** multiclass confusion matrices.

---

## 6. Hybrid outputs and `zero_day` testing

**Rule**

- Stage 1 ≠ Normal → final label = **that attack class**.
- Stage 1 = Normal and score ≤ **T** → **Normal**.
- Stage 1 = Normal and score > **T** → **`zero_day`**.

**Why `zero_day` may be rare**

- If Stage 1 **rarely** misclassifies attacks as Normal, **few windows** enter Stage 2 as attacks.
- Car-Hacking has **only known** classes; there is **no** ground-truth “unknown” column.

**How the script still validates Stage 2**

- **Gated metrics:** among test windows with **Stage 1 = Normal**, binary classification **true attack vs true normal** using score > **T** (Stage 1 **false negatives** as a **proxy** for “suspicious on Normal path”).
- **Synthetic probes:** **Uniform [0,1]^d** random vectors in flattened feature space — **not** real CAN — to check that scores can exceed **T** on out-of-distribution points; optional balanced report vs calibration normals.
- **Diagnostics block:** counts of Stage 2 path, FN fraction above **T**, hybrid **`zero_day`** count, estimated **FPR** on cal normals.

For a **full pseudo–zero-day** protocol (hold out an entire attack class from Stage 1 training), extend the script with a filtered training set and a frozen label space; the summary paper doc describes that at a high level.

---

## 7. Configuration (top of `endsem_new_pipeline.py`)

| Constant | Default | Role |
|----------|---------|------|
| `data_path` | Colab drive or `9) Car-Hacking Dataset` (cwd) | Dataset root |
| `SEQ_LEN` | `12` | Sliding window length (paper-style temporal context) |
| `BATCH_SIZE` | `64` | Stage 1 batch (paper uses 64) |
| `STAGE1_*` | see file | Epochs, patience, LR, weight decay |
| `STAGE2_NORMAL_PERCENTILE` | `97.0` | Higher → stricter **T** → fewer flags |
| `STAGE2_SCORE_AGG` | `"max"` | `"max"` or `"blend"` |
| `AAE_EPOCHS`, `LATENT_DIM`, etc. | see file | Stage 2 capacity and training |

---

## 8. Dependencies

**Python:** `numpy`, `pandas`, `torch`, `scikit-learn`, `matplotlib`, `seaborn`.  

Optional Colab: `google.colab.drive`.

---

## 9. Run

From a directory that contains **`9) Car-Hacking Dataset`** (or edit `data_path`):

```bash
python Endsem_prep/endsem_new_pipeline.py
```

Plots: Stage 1 loss, Stage 1 confusion, Stage 2 AE curve, score histograms, hybrid confusion.

---

## 10. Relation to other prep files

| File | Difference |
|------|----------------|
| `Endsem-imp-pipeline-01.py` | Stage 1 = **CANForge-style** CNN + BiLSTM + SE; **per-row** samples (no sliding window); same Stage 2 idea |
| `Endsem-imp-pipeline-02.py` | Adds **decoder / uniform probes** and alternate Stage 2 score defaults |
| **`endsem_new_pipeline.py`** | Stage 1 = **CANGuard-style** CNN + BiGRU + **attention**; **sliding windows**; flattened-window AAE |

---

*Generated as companion documentation for `Endsem_prep/endsem_new_pipeline.py`.*
