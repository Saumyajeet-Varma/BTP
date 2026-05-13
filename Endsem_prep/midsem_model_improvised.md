# `midsem_model_improvised.py` — two-stage IDS + GAN → `zero_day`

This document matches the current script: a **two-stage pipeline** on Car-Hacking sliding windows, a **GAN** that synthesises fake CAN-like windows, and a **sixth label** `zero_day` for anomalies (including GAN fakes) caught by Stage 2.

---

## 1. Labels (6)

| Label | Meaning |
|--------|---------|
| **Normal** | Benign traffic (ground truth from normal file or `Flag == R` in attack CSVs). |
| **DoS**, **Fuzzy**, **Gear**, **RPM** | Known attacks from the dataset. |
| **zero_day** | Not one of the four known attacks *in the hybrid decision*: either **GAN-generated** windows used only in evaluation, or **true Normal** mis-scored by the autoencoder (false alarm), or rare AE-high cases on real Normal. |

Ground-truth `zero_day` in the script is assigned **only to GAN-generated test windows**. Real Normal / known attacks keep their dataset labels for metrics.

---

## 2. End-to-end pipeline

1. **Load streams** — Same as before: normal text log + four attack CSVs (with alternate filenames). Per-stream time sort and sliding windows (`SEQ_LEN`).

2. **Features** — Base CAN fields plus IAT, ID frequency, byte entropy / sum / range / std.

3. **Scale** — `MinMaxScaler` on flattened window → reshape to `(N, SEQ_LEN, n_features)`; flattened dimension `flat_dim = SEQ_LEN * n_features` for AE and GAN.

4. **Train / test split (real data only)** — 80 / 20 stratified on **five** classes (Normal + four attacks). Stage 1 is trained only on this.

5. **Stage 1 — Known attack classifier (5 classes)**  
   - CNN + stacked LSTM + Dense → **softmax over {Normal, DoS, Fuzzy, Gear, RPM}**.  
   - Class-balanced weights.  
   - **Output:** `midsem_cm_stage1.png` (heatmap) + printed accuracy, precision, recall, F1 (macro / weighted + `classification_report`).

6. **Stage 2 — Autoencoder (anomaly score on “Normal” from training)**  
   - Trained on **flattened windows** that are **true Normal** in the **training** split only (reconstruction targets = input).  
   - **Score:** per-window MSE between input and reconstruction.  
   - **Threshold:** starts at a high percentile of MSE on **train Normal**; then adjusted using **GAN probe** MSEs so a large fraction of synthetic fakes lies above the threshold (hybrid between percentile of real normals and GAN distribution).

7. **GAN — fake normal-like windows**  
   - **Generator:** noise → MLP → `sigmoid` vector of length `flat_dim` in `[0,1]` (same space as scaled real windows).  
   - **Discriminator:** real vs fake on flat vectors.  
   - Trained on **train Normal** flats only (generator tries to fool D while D tries to detect fakes).  
   - After training, **`NUM_GAN_TEST_WINDOWS`** samples are drawn for evaluation with **true label `zero_day`**.

8. **Hybrid inference (final 6-way decision)**  
   - Run **Stage 1** on every window.  
   - If argmax ≠ **Normal** → final label = that **known attack** (Stage 2 is skipped).  
   - If argmax = **Normal** → run **AE** on the flat window: if MSE **>** threshold → **`zero_day`**, else **Normal**.

9. **Evaluation sets**  
   - **Real test windows** (same 5-way labels as data).  
   - **GAN windows** appended with true label **`zero_day`**.  
   - **Final metrics** over **all** of these with fixed label order `FINAL_LABELS` → `midsem_cm_final_hybrid.png`.

10. **Stage 2-only plot (interpretability)**  
    - Subset: samples where **Stage 1 predicted Normal** *and* ground truth is **Normal** or **zero_day** (i.e. GAN).  
    - Binary confusion: predicted **Normal** vs **zero_day** from the AE rule → `midsem_cm_stage2.png`.

---

## 3. What each stage is doing (conceptually)

### Stage 1

Learns discriminative patterns in **short multivariate time series** of CAN features to separate **Normal** from each **known** attack class. It is **not** responsible for novel attacks: anything it confidently calls non-Normal bypasses the AE.

### Stage 2

Learns a **manifold of normal traffic** in flat window space. Windows that **look like Normal to Stage 1** but **do not reconstruct well** are treated as **out-of-distribution** → **`zero_day`**. The GAN provides **synthetic OOD** probes that should trigger high reconstruction error relative to real normals, so the threshold can be validated without a real zero-day dataset.

### GAN

The **generator** does not label data; it **creates challenging negatives** for the AE. If the GAN matches the normal manifold too well, MSE may stay low (harder `zero_day` detection); weaker GANs often yield **higher** AE error and are **easier** to flag.

---

## 4. Output artefacts (same directory as the script)

| File | Content |
|------|---------|
| `midsem_cm_stage1.png` | Confusion matrix heatmap — Stage 1 only, 5 classes. |
| `midsem_cm_stage2.png` | Heatmap — AE decision when Stage 1 says Normal; truth Normal vs `zero_day`. |
| `midsem_cm_final_hybrid.png` | Heatmap — full hybrid, 6 classes. |

Each heatmap title includes **accuracy** and **macro precision / recall / F1**. The console prints the same plus **weighted** averages and sklearn **`classification_report`**.

---

## 5. Hyperparameters (top of `.py`)

Key knobs: `SEQ_LEN`, `STAGE1_EPOCHS`, `AE_EPOCHS`, `GAN_EPOCHS`, `GAN_STEPS_PER_EPOCH`, `NUM_GAN_TEST_WINDOWS`, `STAGE2_NORMAL_PERCENTILE`, `GAN_NOISE_DIM`, batch sizes.

---

## 6. How to run

Set `data_path` to the Car-Hacking folder, install TensorFlow + sklearn + pandas + numpy + matplotlib + seaborn, then:

`python midsem_model_improvised.py`

If the dataset path is missing, the script exits with an error before training.
