# `midsem_second_stage_imp.py` — improved Stage 2 anomaly detection

This file is a **variant** of the hybrid pipeline in `midsem_model_improvised.py`: **Stage 1** (5-class known attacks + Normal) and **GAN** synthetic probes are the same idea; **Stage 2** is redesigned to address weak **`zero_day` recall** when anomaly detection used **only reconstruction MSE** and a **hand-tuned** threshold.

---

## 1. What stays the same

- **Data:** Car-Hacking streams, sliding windows, engineered features, `MinMaxScaler`, stratified splits.  
- **Stage 1:** CNN + LSTM, five classes: Normal, DoS, Fuzzy, Gear, RPM.  
- **GAN:** Unconditional generator in flattened `[0,1]` window space; discriminator; trained on **fit-split** normal flats only.  
- **Hybrid rule:** If Stage 1 ≠ Normal → that attack; if Stage 1 = Normal → Stage 2 score vs threshold → Normal or **`zero_day`**.  
- **Test evaluation:** Real held-out test windows + GAN windows with true label **`zero_day`**.  
- **Figures:** `midsecond_cm_stage1.png`, `midsecond_cm_stage2.png`, `midsecond_cm_final_hybrid.png` (green colormap to distinguish from the original script’s blues).

---

## 2. What changes — improved Stage 2

### 2.1 Bottleneck encoder–decoder

Instead of a single symmetric MLP autoencoder without an explicit bottleneck **latent vector**, this script builds:

- **`encoder`:** `flat_dim` → … → **`z`** with dimension **`AE_LATENT_DIM`** (default 40).  
- **Full `ae`:** same path through `z` then decode back to `flat_dim` with **sigmoid** outputs.

Training minimizes **MSE** between input and reconstruction on **true Normal** windows from the **inner training fit** split only (no leakage from validation or test).

The **encoder output `z`** is used as an extra signal: out-of-distribution / GAN samples often deviate in **latent norm** as well as in **reconstruction error**.

### 2.2 Combined anomaly score (not MSE alone)

On **train-normal** windows only, the script estimates:

- \(\mu_\text{mse}, \sigma_\text{mse}\) — mean and std of per-window **MSE** \((x - \hat{x})^2\) (mean over features).  
- \(\mu_\ell, \sigma_\ell\) — mean and std of **\(\|z\|_2\)** (L2 norm of the latent).

For any window, define z-scores:

\[
z_\text{mse} = \frac{| \text{MSE} - \mu_\text{mse} |}{\sigma_\text{mse}}, \quad
z_\ell = \frac{| \|z\|_2 - \mu_\ell |}{\sigma_\ell}
\]

**Anomaly score** (same spirit as combined recon/latent scores in your `endsem_new_pipeline.py`):

\[
\text{score} = \max(z_\text{mse}, z_\ell)
\]

So a sample is flagged if **either** reconstruction is unusual **or** the latent embedding is unusual relative to train-normal statistics.

### 2.3 Train / validation / test splits (why)

Original pipeline: **one** training split → AE threshold from **train-normal percentiles + GAN probe heuristics**, then test. That can mis-calibrate when **MSE and GAN** distributions overlap.

Here:

1. **Outer:** 80% `train_big` / 20% `test` (stratified, five classes).  
2. **Inner:** `train_big` → **~68% `X_fit`** / **~12% `X_val`** (stratified).  
   - **Stage 1** trains on **`X_fit`**.  
   - **AE** trains on **Normal rows inside `X_fit`**.  
   - **GAN** trains on those same normal flats.

3. **Threshold tuning** uses **`X_val`** (never seen during AE weight updates):
   - **Negatives:** combined scores on **true Normal** windows in validation where **Stage 1 also predicts Normal** (if too few, relax to all val rows with Stage 1 = Normal).  
   - **Positives:** fresh **GAN** windows; keep only those where **Stage 1 predicts Normal** (so they actually reach Stage 2 in deployment). If too few, use all GAN scores.

4. **Binary labels** for tuning: Normal = 0, synthetic anomaly (GAN) = 1.  
5. **Grid search:** `THRESHOLD_GRID_POINTS` (default 400) evenly spaced scores between min and max of the **concatenated** neg+pos score vectors; pick **`T`** that maximizes **F1** on that validation set.

6. **Test + final hybrid** use this **`T`** on the held-out **test** set plus **test GAN** windows (same protocol as the original improvised script).

This directly optimizes the **Normal vs `zero_day`** trade-off on data that was **not** used to fit the AE weights.

---

## 3. Hyperparameters (file header)

| Symbol | Role |
|--------|------|
| `AE_LATENT_DIM` | Bottleneck size for `z`. |
| `NUM_GAN_VAL_PROBES` | GAN samples used **only** for threshold F1 tuning on val. |
| `THRESHOLD_GRID_POINTS` | Resolution of the score grid for choosing `T`. |
| `GAN_*` | GAN training length / batch (slightly increased vs baseline for stabler fakes). |

---

## 4. Expected effect vs baseline

- **MSE-only** thresholds often miss GANs that still reconstruct “okay” but sit off-manifold in **`z`**.  
- **Max(z_mse, z_lat)** catches more of those → higher **`zero_day` recall** possible, with **F1 tuning** controlling **Normal** false alarms.

Exact numbers depend on run (random GAN, TF). Compare **`midsecond_cm_stage2.png`** to the original **`midsem_cm_stage2.png`**.

---

## 5. How to run

Same dataset path as other midsem scripts. From the `Endsem_prep` folder:

`python midsem_second_stage_imp.py`

Requires TensorFlow, scikit-learn, pandas, numpy, matplotlib, seaborn.

---

## 6. Relation to `midsem_model_improvised.py`

| Piece | Original | This file |
|--------|----------|-----------|
| AE | Symmetric MLP, MSE only | Bottleneck + explicit **`z`**, score uses **MSE + \(\|z\|\)** |
| Threshold | Percentile + GAN heuristic | **Validation F1** on Normal vs GAN (Stage-1-Normal slice) |
| Train split | Single train for S1+AE | **Inner val** reserved for **threshold only** |

You can keep both scripts: baseline vs improved Stage 2 ablation.
