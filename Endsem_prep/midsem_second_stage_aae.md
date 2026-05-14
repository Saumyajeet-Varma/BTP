# `midsem_second_stage_aae.py` — Stage 2 as Adversarial Autoencoder (AAE)

This script follows **`midsem_second_stage_imp.py`** for data, splits, **Stage 1** (5-class CNN–LSTM), **flat-space GAN** probes for `zero_day`, and **val F1** thresholding on Stage-2 scores. Only **Stage 2** is replaced by an **Adversarial Autoencoder** in latent space.

---

## 1. Architecture (Stage 2)

| Module | Input → output | Role |
|--------|----------------|------|
| **Encoder** `E` | flattened window `[0,1]^{d}` → `z ∈ ℝ^{L}` (`L` = `AAE_LATENT_DIM`, linear head) | Map traffic window to latent code. |
| **Decoder** `G` | `z` → reconstruction `\hat{x}` (sigmoid, same `d`) | Minimize **MSE** `\|x - \hat{x}\|^2` on **train-Normal** flats (reconstruction error). |
| **Latent discriminator** `D_z` | `z` → scalar in `(0,1)` | **Real:** `z ∼ \mathcal{N}(0, I)`. **Fake:** `z = E(x)` with `x` normal. Trained to assign **1** to Gaussian draws and **0** to encoded normals. |

**Adversarial phase:** each step (see code):

1. **Update `D_z`:** minimize BCE on `(z_real, 1)` and `(E(x), 0)` with `E(x)` **stop-gradient** so only `D_z` learns.
2. **Update `E` and `G`:** minimize `MSE(x, G(E(x))) + λ · BCE(D_z(E(x)), 1)` with **`D_z` frozen** so the encoder pushes `z` toward the Gaussian prior while keeping recon low (`λ` = `AAE_ADV_WEIGHT`).

This matches the usual **AAE** idea: recon binds `E,G` to the data manifold; the latent **GAN** term regularizes `z` toward `N(0,I)`.

---

## 2. Training schedule

1. **Pretrain** the chained `autoencoder = G(E(·))` with **MSE only** on `X_fit_normal_flat` (`AAE_PRETRAIN_EPOCHS`, early stopping `AAE_PRETRAIN_PATIENCE`) — same spirit as fitting the vanilla AE in `midsem_second_stage_imp.py` before using extra signals.
2. **Adversarial** loop: `AAE_ADV_EPOCHS` × `AAE_STEPS_PER_EPOCH` minibatches from the same normals (`train_aae_adversarial`).
3. **GAN** in **flat** space (unchanged from `imp`) for synthetic `zero_day` at val/test.

---

## 3. Anomaly score and threshold (aligned with `imp`)

On **train-normal** windows only, estimate:

- `μ_mse, σ_mse` — mean/std of per-window **reconstruction MSE**.
- `μ_D, σ_D` — mean/std of **`D_z(E(x))`** (discriminator on encoded normal).

For any window:

\[
\text{score} = \max\left( \frac{| \text{MSE} - \mu_\text{mse} |}{\sigma_\text{mse}}},\; \frac{| D_z(E(x)) - \mu_D |}{\sigma_D} \right)
\]

So anomalies can show up as **large recon error** and/or **latent-discriminator deviation** (replacing `||z||` in the reference script).

**Threshold `T`:** same protocol as `midsem_second_stage_imp.py` — grid on val **Normal vs GAN**, Stage 1 = Normal, maximize **F1**.

---

## 4. Hyperparameters (header)

| Name | Default | Notes |
|------|---------|--------|
| `AAE_LATENT_DIM` | 40 | Latent size (matches `AE_LATENT_DIM` in `imp`). |
| `AAE_PRETRAIN_EPOCHS` | 12 | Recon-only warmup. |
| `AAE_ADV_EPOCHS` / `AAE_STEPS_PER_EPOCH` | 35 / 130 | Adversarial schedule (same step count style as flat GAN in `imp`). |
| `AAE_BATCH` | 256 | Minibatch for AAE steps. |
| `AAE_ADV_WEIGHT` | 0.05 | Trade-off recon vs fooling `D_z`. |

GAN and Stage-1 constants match **`midsem_second_stage_imp.py`** unless you change them locally.

---

## 5. Outputs

- `midsecond_aae_cm_stage1.png`  
- `midsecond_aae_cm_stage2.png`  
- `midsecond_aae_cm_final_hybrid.png`  

Run the script the same way as `midsem_second_stage_imp.py` (local `data_path` or Colab drive at the top of the file).

---

## 6. Note on “discriminator according to reconstruction error”

The **latent discriminator** here does **not** take reconstruction error as an input; it sees **`z` only**, while **reconstruction** is optimized in parallel via the **decoder** and included in the **anomaly score** through the **MSE** branch (same two-branch `max` structure as `imp`, second branch is **`D_z(E(x))`** instead of **`||z||`**). If you want **`D_z([z, mse])`**, that would be a small architectural extension on top of this file.
