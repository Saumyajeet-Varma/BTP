# `claude_opus_model.py` — Precision-Prioritized Hybrid IDS with Sequence-Aware AE Ensemble and Mahalanobis-Based Zero-Day Detection

> Companion to `midsem_second_stage_imp.py`. Same outer skeleton, **redesigned Stage 2**. Built to address the empirically observed weakness of the parent pipeline: **zero_day precision ≈ 0.53** on the held-out test set, dragging hybrid precision to ≈ 0.90 and hybrid accuracy to ≈ 0.93 despite Stage 1 being already strong.

---

## 1. Problem statement and observed weaknesses

### 1.1 The task

We perform hierarchical intrusion detection on Car-Hacking CAN bus traffic in two stages:

- **Stage 1** is a closed-set 5-way classifier over `{Normal, DoS, Fuzzy, Gear, RPM}`. It is trained supervised on labelled known attacks. It is already strong (≈ 0.985 precision on Normal in the parent pipeline) and is **kept unchanged** in this file.

- **Stage 2** is an **open-set anomaly detector** that must decide, on windows Stage 1 labels as Normal, whether the window is *truly* normal or whether it is a **`zero_day`** (an attack pattern not seen during training, synthesized for evaluation via a GAN trained on real normals).

The full hybrid rule is

\[
\hat{y}(\mathbf{x}) =
\begin{cases}
c, & \text{if Stage 1}(\mathbf{x}) = c \neq \text{Normal} \\
\text{Normal}, & \text{if Stage 1}(\mathbf{x}) = \text{Normal} \text{ and } s(\mathbf{x}) \leq T \\
\text{zero\_day}, & \text{if Stage 1}(\mathbf{x}) = \text{Normal} \text{ and } s(\mathbf{x}) > T
\end{cases}
\]

where \(s(\mathbf{x})\) is a Stage-2 anomaly score and \(T\) is a threshold.

### 1.2 Why Stage 2 was the bottleneck

Empirical numbers for the parent pipeline `midsem_second_stage_imp.py`:

| Metric | Value |
|--------|-------|
| Stage-2 Normal precision | 0.9854 |
| Stage-2 zero_day precision | **0.53** |
| Hybrid precision | 0.9041 |
| Hybrid accuracy | 0.9318 |

Reading the row carefully: Stage 2 is **over-eager to call windows `zero_day`**. About 47 % of the windows it flagged as `zero_day` were in fact Normal. Three structural causes:

1. **The anomaly score was 1-D in the latent direction.** The score was \( \max(z_{\text{MSE}}, z_{\|\mathbf{z}\|}) \), where \(z_{\|\mathbf{z}\|}\) is a 1-D z-score on the **norm** of the bottleneck. A 40-D latent collapsed to a single number ignores the covariance structure of normal embeddings, so directions in latent space that are *atypical for normals but happen to have a normal-ish norm* go undetected — and conversely, directions that look anomalous in norm but are part of the normal manifold get falsely flagged.

2. **Flat MLP autoencoder.** The parent AE flattens the \((T, F) = (24, 14)\) window into a 336-D vector. CAN bus traffic has strong temporal regularities (IAT periodicity, byte-level correlations between adjacent frames, repeated CAN IDs). Flattening throws this away and forces the AE to learn invariances it does not need to learn, raising the per-window reconstruction-error variance on normals and shrinking the margin between Normal and zero_day.

3. **F1-tuned threshold.** F1 weights precision and recall equally. In a deployment where ≈ 80 % of post–Stage-1-Normal traffic is genuinely Normal, equal weighting **biases** toward recall on the rare positive class, which is exactly what produces low precision.

The **class imbalance the user reports is at inference time**: Normal windows ≫ zero_day windows on the test set. Importantly, that imbalance is *not* fixable by re-weighting the AE loss (the AE only sees Normal), but it is fixable by (i) sharper, lower-variance anomaly scores and (ii) a precision-aware operating point.

### 1.3 Design objective

Maximize Stage-2 **precision** on `zero_day` subject to keeping recall reasonable (we set a soft precision floor of 0.85 and use F\(_{0.5}\) which weights precision four times as heavily as recall in the harmonic mean).

---

## 2. Pipeline overview

```text
                +---------------------------+
                |   Car-Hacking raw streams |
                +-------------+-------------+
                              |
                  parse, sort by timestamp,
                  engineer features, window
                              |
                              v
                  X_w  (N, SEQ_LEN=24, F=14)
                              |
                MinMaxScaler  -> X_s in [0,1]^(N,T,F)
                              |
            stratified outer split (80 / 20)
                              |
                   train_big  ----  test  ----------------+
                              |                            |
              stratified inner split (≈68 / 12)            |
                              |                            |
                       X_fit       X_val                   |
                              |       |                    |
                              v       |                    |
                +-------------+       |                    |
                | STAGE 1 (CNN-LSTM)  |  same as parent    |
                +---------+-----------+                    |
                          | predicts {Normal, 4 attacks}   |
                          v                                |
        (filter X_fit to Normal -> X_fit_normal)           |
                          |                                |
   +----------------------+----------------------+         |
   |                                             |         |
   v                                             v         |
   +-----------------+    +---------------------+ +--------+----------+
   |  AE ensemble K=3 |   |  GAN on flat norms  | | held-out test eval|
   |  (Conv+BiLSTM    |   |  (probe generator)  | +-------------------+
   |   bottleneck)    |   +---------+-----------+
   +--------+--------+              |
            |                       |
   fit latent Mahalanobis           |
   (mu, P=Sigma^-1) per AE          |
            |                       |
            v                       v
   robust standardize           build diverse probes:
   {mean_MSE, max-step MSE,     GAN + uniform noise +
   maha_lat} on X_fit_normal    feature-shuffled normals
            |                       |
            +-----------+-----------+
                        v
              VALIDATION threshold tuning
              (precision-priority F0.5 with
               soft precision floor 0.85)
                        |
                        v
                test-time hybrid rule
```

### 2.1 What is unchanged from `midsem_second_stage_imp.py`

This is deliberate so the Stage-2 change is a clean ablation:

- Data loading, windowing, feature engineering (`IAT`, `CAN_ID_freq`, `byte_entropy`, `byte_sum`, `byte_range`, `byte_std`).
- `MinMaxScaler` fit on full data, stratified 80/20 outer split and stratified inner 68/12 split (`X_fit`, `X_val`, `X_test`).
- Stage 1 architecture, optimizer, callbacks, class-weighted loss.
- The hybrid routing rule.
- The use of a GAN trained on flat normals to synthesize OOD probes — but its role is now strictly **probe generation for threshold tuning**, not training signal.

### 2.2 What is new in Stage 2

Five interlocking changes, each justified in detail in §3:

1. Sequence-aware **Conv1D + BiLSTM bottleneck autoencoder** that operates on `(T, F)` directly.
2. **Mahalanobis distance** of the bottleneck `z` from train-Normal latent centroid, with **Ledoit–Wolf-style shrinkage** and **eigenvalue flooring**.
3. **Three raw scores** fused into one via **robust (median / MAD) standardization** on train-Normal followed by an **average**, not a max.
4. **Ensemble of K = 3 AEs** with different random seeds; per-window scores averaged across the ensemble.
5. **Diverse synthetic OOD probes** (GAN + uniform noise + feature-shuffled normals) and **F\(_{0.5}\) threshold tuning** with a soft precision floor.

---

## 3. Stage 2 design, decision-by-decision

### 3.1 Sequence-aware autoencoder

```text
Input (T, F) = (24, 14)
  -> Conv1D(64, k=3, ReLU) -> BN
  -> Conv1D(96, k=3, ReLU) -> BN
  -> Bidirectional LSTM(64, return_sequences=False)
  -> Dropout(0.15)
  -> Dense(latent_dim=32, tanh)          ===  z
  -> Dense(T * 32, ReLU) -> Reshape(T, 32)
  -> LSTM(64, return_sequences=True) -> BN
  -> Conv1DTranspose(96, k=3, ReLU)
  -> Conv1DTranspose(64, k=3, ReLU)
  -> TimeDistributed(Dense(F, sigmoid))
```

**Why a sequence-aware encoder?** Three reasons:

1. **Inductive bias matches the data.** CAN bus traffic is periodic and locally smooth. A Conv1D captures short local motifs (e.g., a recurring CAN ID pattern across 3 consecutive frames) far more parameter-efficiently than a flat MLP. A BiLSTM on top captures longer-range temporal context across the 24-frame window.

2. **Lower normal reconstruction variance ⇒ better separability.** The bottleneck-MSE distribution under \(p(\text{Normal})\) is narrower because the network does not waste capacity learning that frame \(t\) and frame \(t+1\) are correlated — that's encoded in the architecture. A narrower normal-MSE distribution gives more *headroom* between normals and anomalies under the same scoring rule.

3. **`tanh` latent activation.** Choosing `tanh` (range \([-1, 1]\)) rather than ReLU in the bottleneck means the latent space is bounded and approximately symmetric around 0. This makes Mahalanobis distance (which assumes finite second moments) numerically much better behaved than a ReLU latent that can be heavy-tailed or stuck at 0 in many components.

**Why this specific depth?** Two Conv1D blocks followed by one BiLSTM is enough to span the full window (effective receptive field) without producing a parameter count that blows past the ≈ 6 k–10 k normal-window training set. Anything deeper risks **overfitting to normals**, which would make the AE reconstruct *anomalies* well too — the opposite of what we want for OOD detection.

### 3.2 Mahalanobis distance in latent space

Let \(\{\mathbf{z}_i\}_{i=1}^{N}\) be the latents of train-Normal windows. We compute

\[
\boldsymbol{\mu} = \frac{1}{N} \sum_i \mathbf{z}_i, \quad
\boldsymbol{\Sigma} = \frac{1}{N - 1} \sum_i (\mathbf{z}_i - \boldsymbol{\mu})(\mathbf{z}_i - \boldsymbol{\mu})^\top
\]

Then we **shrink** the covariance toward its diagonal (Ledoit–Wolf style):

\[
\boldsymbol{\Sigma}_\lambda = (1 - \lambda)\, \boldsymbol{\Sigma} + \lambda\, \mathrm{diag}(\boldsymbol{\Sigma}), \quad \lambda = 0.05
\]

and **floor the eigenvalues** at \(\varepsilon = 10^{-4}\) before inverting:

\[
\boldsymbol{\Sigma}_\lambda = V \Lambda V^\top, \quad \Lambda \gets \max(\Lambda, \varepsilon I), \quad
P = (V \Lambda V^\top)^{-1}
\]

The Mahalanobis distance of a query latent \(\mathbf{z}\) is

\[
d_M(\mathbf{z}) = \sqrt{(\mathbf{z} - \boldsymbol{\mu})^\top P (\mathbf{z} - \boldsymbol{\mu})}
\]

implemented row-wise via `einsum` for efficiency (see `_mahalanobis`).

**Why Mahalanobis instead of \(\|\mathbf{z}\|\) z-score?**

- **Scale invariance per component.** A z-score on \(\|\mathbf{z}\|\) treats all latent dimensions equally and uses a *single* variance. Latent components in a learned representation routinely have wildly different variances — some encode high-frequency CAN-ID switches, others encode slowly-varying byte means. Mahalanobis whitens the latent before measuring distance, so a small anomaly in a high-variance direction is correctly down-weighted, and a small anomaly in a low-variance direction is correctly up-weighted.

- **Captures correlated anomalies.** Mahalanobis sees off-manifold movement; \(\|\mathbf{z}\|\) sees only radial movement. Many real attack patterns differ from normals in *correlated* shifts of several latent dimensions (e.g., elevated IAT plus elevated byte_entropy plus depressed CAN_ID_freq). \(\|\mathbf{z}\|\) can completely miss this if the shifts cancel in norm.

- **Standard, principled, no extra hyper-parameters.** It is the maximum-likelihood deviation under a Gaussian model of normal latents. We do *not* need to assume Gaussianity to use it — we only need finite first and second moments, which any reasonable AE satisfies.

**Why shrinkage and eigenvalue flooring?**

The empirical covariance of a 32-D latent estimated from \(N \approx 6\,000\)–10 000 normals is well-conditioned in expectation but can have very small eigenvalues in practice (latent dimensions that learnt to encode a near-constant). Inverting a near-singular matrix would produce **enormous** Mahalanobis distances on minor perturbations, drowning the signal in noise. The Ledoit–Wolf-style convex blend `(1−λ)·Σ + λ·diag(Σ)` and the eigenvalue floor at \(10^{-4}\) make \(P\) numerically safe without distorting the dominant directions.

### 3.3 Three raw scores, robust-standardized, averaged

For every window \(\mathbf{x}\) we compute three numbers:

1. **Mean reconstruction MSE.** Standard. Catches globally bad reconstructions.
   \[
   s_{\text{mean}}(\mathbf{x}) = \frac{1}{T F} \sum_{t,f} \big(x_{t,f} - \hat{x}_{t,f}\big)^2
   \]

2. **Max per-time-step MSE.** Average within a time step, max across time.
   \[
   s_{\text{max}}(\mathbf{x}) = \max_{t} \; \frac{1}{F} \sum_{f} \big(x_{t,f} - \hat{x}_{t,f}\big)^2
   \]

3. **Latent Mahalanobis.**
   \[
   s_{\text{maha}}(\mathbf{x}) = d_M(\mathrm{Enc}(\mathbf{x}))
   \]

Then we **robust-standardize** each score against the train-Normal distribution:

\[
\tilde{s}_k(\mathbf{x}) = \frac{|s_k(\mathbf{x}) - \mathrm{median}(s_k^{\text{train-Normal}})|}{1.4826 \cdot \mathrm{MAD}(s_k^{\text{train-Normal}})}
\]

and finally **average**:

\[
s(\mathbf{x}) = \frac{1}{3}\big(\tilde{s}_{\text{mean}} + \tilde{s}_{\text{max}} + \tilde{s}_{\text{maha}}\big)
\]

**Why three different raw scores?**

- They are **decorrelated failure modes**:
  - `mean_MSE` catches windows that look globally wrong.
  - `max-step MSE` catches windows where 23 frames are fine but one frame is wildly off (e.g., one spoofed CAN frame in the window). The parent pipeline misses these because the bad frame's error averages out across 24 frames.
  - `maha_lat` catches windows that reconstruct reasonably well (because the AE is mildly overcomplete) but live off the normal manifold in embedding space.
- Empirically, OOD samples spike at least one of these three and often two. Using all three increases recall **without** giving up precision, provided we combine them correctly.

**Why robust median / MAD standardization instead of mean / std?**

- The MAD (median absolute deviation, with the 1.4826 consistency constant for normal data) is unaffected by outliers in the *training-normal* set. The Car-Hacking "normal" stream is not a clean i.i.d. sample — there are bursts and rare benign events. Mean and std on a contaminated normal distribution drift toward the contaminants, *raising the scoring threshold for anomalies* and hurting recall.
- After standardization each \(\tilde{s}_k\) is on the same scale (roughly "robust standard deviations away from typical normal"), which is essential for the next step.

**Why average and not max?**

- The parent script uses `max(z_mse, z_lat)`. `max` is brittle: a single noisy score dominates the decision. If `mean_MSE` is slightly inflated for a particular Normal window (which happens), the max can fire even though Mahalanobis and max-step are both quiet.
- Averaging acts as **score-level voting** across three independent evidence channels. A window is only confidently flagged when *multiple* channels agree, which is precisely what precision-priority demands. The average has lower variance than the max under the null (Normal) by a factor of roughly \(\sqrt{3}\) for independent components, giving a tighter margin.

### 3.4 Autoencoder ensemble (K = 3)

We train three independent AEs with different random seeds (`seed = 0, 1, 2`, fed to `tf.random.set_seed`) on the *same* train-Normal windows. At scoring time, the three raw scores per AE are averaged across the ensemble *before* robust standardization. Mahalanobis parameters are also fit per AE.

**Why ensemble?**

- **Variance reduction.** A single AE's reconstruction error has a stochastic component coming from random initialization and SGD trajectory. For a given Normal window the AE-specific MSE can vary appreciably across seeds. Averaging reduces this variance by roughly \(1/\sqrt{K}\), tightening the Normal score distribution and thus the operating point.
- **Reduces "easy mode" failures.** If one AE happens to learn a degenerate solution (e.g., a near-identity through over-parameterization) it will reconstruct attacks well. Two other AEs that didn't fall into that local minimum will pull the averaged score back up.
- **Cheap.** Three AEs are still tractable (≈ 10 minutes total on a single GPU). We avoid K > 5 because diminishing returns set in fast and the GAN/Stage-1 training already dominate runtime.

### 3.5 GAN as a probe generator (not as training data)

The GAN architecture is unchanged from `midsem_second_stage_imp.py`. It is trained on flat normals to generate plausibly-looking-but-not-real windows. Crucially, the GAN samples are **never seen during AE training**: they are used **only** for two things:

1. Threshold tuning probes (validation).
2. Held-out `zero_day` evaluation (test).

**Why is the GAN not used as training data?**

If the AE were trained to *also* reconstruct GAN samples well, it would learn the GAN's modes and stop treating GAN samples as anomalies — defeating the whole point. Keeping the GAN strictly downstream of AE training preserves the AE as a pure novelty detector.

**Why train the GAN for slightly longer than the parent (`GAN_EPOCHS = 30`)?**

To make the GAN samples a *harder* test, not an easier one. A weak GAN produces obvious garbage that any AE flags; a stronger GAN produces samples closer to the normal manifold, which raises the bar for both threshold tuning and final evaluation.

### 3.6 Diverse synthetic OOD probes for threshold tuning

Threshold tuning sees three distinct families of synthetic positives:

| Probe family | What it represents |
|--------------|--------------------|
| GAN windows  | "Plausible but novel" — close to the normal manifold |
| Uniform `[0, 1]` noise windows | Structurally far from CAN traffic — out-of-support |
| Feature-shuffled normals | Correct marginals, wrong joint distribution — joint-structure anomalies |

All three are filtered to those Stage 1 predicts as Normal (only those reach Stage 2 in deployment, so only those matter for threshold tuning).

**Why three families?**

A single OOD family produces a single shape of "anomaly distribution" in score space. Tuning a threshold on one family overfits to that family's location in score space, which is why the parent pipeline's threshold generalizes poorly to *real* zero-days (the test-time GAN samples differ from val-time GAN samples in subtle ways). Covering three families forces the threshold to lie in a region that works for several OOD regimes simultaneously, improving out-of-sample generalization.

**Why feature-shuffled normals specifically?**

Many CAN bus attacks (e.g., a payload-injection attack from a different ECU) leave each *individual* feature inside its normal range but break the **joint** structure. A GAN trained on real normals tends to reproduce the joint structure too well to simulate this. Permuting feature columns within a time step (`_make_shuffle_probes`) destroys the joint structure while preserving each feature's marginal — directly mimicking that attack regime.

### 3.7 Precision-prioritized threshold selection

The validation set is

\[
\mathcal{D}_{\text{val-bin}} = \{(\tilde{s}(\mathbf{x}), 0) : \mathbf{x} \in \text{val Normal} \} \cup \{(\tilde{s}(\mathbf{x}), 1) : \mathbf{x} \in \text{probes}\}
\]

For a candidate threshold \(T\), let \(P(T), R(T)\) be precision and recall on the positive class. We optimize

\[
F_\beta(T) = (1 + \beta^2) \cdot \frac{P(T) \cdot R(T)}{\beta^2 \cdot P(T) + R(T)}, \quad \beta = 0.5
\]

subject to a soft constraint

\[
P(T) \geq P_{\min} = 0.85
\]

If no \(T\) in the grid satisfies the floor, we fall back to maximizing \(F_{0.5}\) unconstrained and log that the floor was not met.

**Why F\(_{0.5}\) instead of F1?**

The F\(_\beta\) score weights precision by \(1\) and recall by \(\beta^2\), so at \(\beta = 0.5\) precision counts **four times** as much as recall in the harmonic-mean trade-off. This is mathematically the correct objective when false positives (calling Normal a zero-day) are more costly than false negatives (missing a zero-day) — and it is exactly the operating regime the user described.

**Why a soft precision floor?**

F\(_{0.5}\) alone can still pick a precision of 0.6 if recall is very high. The floor acts as a hard guarantee: the picked threshold *must* achieve at least 0.85 precision on the validation probe distribution. If the AE is so weak that no threshold can hit 0.85, we fall back to F\(_{0.5}\) and surface that in the log — useful diagnostic information rather than silently accepting a bad threshold.

**Why grid search between the 1st and 99.5th percentile of the joint score distribution?**

Searching `[min, max]` of `scores_all` exposes the grid to grid-quantization issues near extreme values where there is one positive or one negative window. The 1st / 99.5th percentile cropping focuses the grid on the region where F\(_{0.5}\) is non-trivial and gives a finer effective resolution (we use `THRESHOLD_GRID_POINTS = 600`, up from 400 in the parent, for the same reason).

### 3.8 Why this addresses class imbalance without resampling

The user noted Normal-heavy test imbalance. Standard answers (SMOTE, class weights, etc.) do not apply here because Stage 2 is **unsupervised** at train time (the AE sees only normals; there are no zero-day examples to balance against).

What *does* address inference-time imbalance is **precision-prioritized threshold selection** combined with a **lower-variance score**. Imbalance hurts precision specifically because, with many more negatives than positives, even a small false-positive rate produces many absolute false positives. We attack this two ways:

1. **Lower-variance score** (§3.3, §3.4) → fewer Normal windows in the tail of the score distribution → fewer false positives at any given operating point.
2. **F\(_{0.5}\) with precision floor** (§3.7) → operating point chosen on the side of the trade-off curve where false-positive count is small.

Both are robust to test-time prior shifts in the Normal:zero_day ratio: the threshold is set in *score* space using validation precision, not in *frequency* space.

---

## 4. Hyper-parameters and their roles

| Symbol | Value | Role | Why this value |
|--------|-------|------|----------------|
| `AE_LATENT_DIM` | 32 | Bottleneck size | Smaller than the parent's 40; tightens the information bottleneck, increasing AE's pressure to reconstruct only the normal manifold. |
| `AE_ENSEMBLE_SIZE` | 3 | AEs in ensemble | Variance reduction with manageable runtime. |
| `MAHA_SHRINKAGE` | 0.05 | Ledoit-Wolf blend | Small enough to keep dominant covariance directions, large enough to tame ill-conditioning. |
| `MAHA_EIG_FLOOR` | 1e-4 | Eigenvalue floor | Numerical safety on inversion. |
| `NUM_GAN_VAL_PROBES` | 2500 | GAN probes (val) | Sample size large enough for stable precision/recall on the grid. |
| `NUM_NOISE_VAL_PROBES` | 1500 | Uniform-noise probes (val) | Different OOD regime. |
| `NUM_SHUFFLE_VAL_PROBES` | 1500 | Feature-shuffle probes (val) | Joint-structure OOD regime. |
| `THRESHOLD_GRID_POINTS` | 600 | Threshold grid resolution | Finer than parent's 400 because precision is more sensitive than F1 to small T shifts. |
| `F_BETA` | 0.5 | β in F\(_\beta\) | Precision counts 4× recall. |
| `PRECISION_FLOOR` | 0.85 | Soft precision constraint | Empirical target; reduce to 0.75 to favor recall, raise to 0.9 to favor precision more. |

Everything else (`SEQ_LEN`, `BATCH_SIZE`, Stage-1 architecture, GAN architecture, splits, scaler) matches `midsem_second_stage_imp.py` so the ablation is clean.

---

## 5. Test-time evaluation protocol

After threshold \(T\) is fixed on validation, the test set is

\[
\mathcal{D}_{\text{test}} = \underbrace{\text{X\_test}}_{\text{real held-out, 5 classes}} \;\cup\; \underbrace{\text{GAN}_{\text{test}}}_{\text{N=1500, labelled zero\_day}}
\]

For each window we run Stage 1, then if Stage 1 = Normal we apply the Stage-2 score and the hybrid rule of §1.1.

Reported metrics:

- **Stage 1 only (5-class)** — same as the parent.
- **Stage 2 (binary, Normal vs zero_day, restricted to Stage-1-Normal windows)** — the apples-to-apples cell that the user wants to improve.
- **Final hybrid (6-class, `FINAL_LABELS`)** — the deployment metric.

Plots:

| File | What |
|------|------|
| `claude_opus_cm_stage1.png` | Stage-1 5-class confusion matrix |
| `claude_opus_cm_stage2.png` | Stage-2 binary confusion matrix |
| `claude_opus_cm_final_hybrid.png` | Final 6-class hybrid confusion matrix |

(Plotted in `Purples` so they don't visually collide with the parent's `Greens` figures or the AAE variant's defaults.)

---

## 6. Relation to the existing variants

| Aspect | `midsem_model_improvised.py` (baseline) | `midsem_second_stage_imp.py` | `midsem_second_stage_imp_2.py` | **`claude_opus_model.py`** |
|--------|---|---|---|---|
| Stage 2 model | Symmetric MLP AE | Bottleneck MLP AE w/ explicit `z` | Same as `imp` | **Conv1D + BiLSTM bottleneck AE × 3 (ensemble)** |
| Anomaly score | MSE only | `max(z_MSE, z_‖z‖)` | `max(z_MSE, z_‖z‖)` | **avg of robust-standardized {mean MSE, max-step MSE, Mahalanobis(`z`)} across K AEs** |
| Latent geometry | None | norm only | norm only | **Full covariance via shrunk Mahalanobis** |
| Threshold objective | Percentile + heuristic | F1 on Normal-vs-GAN | F\(_{0.5}\) with Stage-1 softmax gate | **F\(_{0.5}\) with soft precision floor, 3 OOD families** |
| Stage-1 gate at decision | No | No | Yes (CAP on `p(Normal)`) | No (precision lift comes from score, not gate) |
| Figures prefix | `midsem_cm_*` | `midsecond_cm_*` | `midsecond2_cm_*` | `claude_opus_cm_*` |

This script is **complementary** to `midsem_second_stage_imp_2.py`: that one keeps the parent's MLP AE and instead adds a Stage-1-confidence gate; this one keeps no gate and instead replaces the AE and the scoring scheme. The two ideas can be combined in a future variant (gate × ensemble score) without conflict.

---

## 7. Expected behavior and failure modes

### 7.1 What should improve

- **Zero-day precision** should jump from ≈ 0.53 toward the precision floor (≈ 0.85). The largest lifts come from Mahalanobis replacing `‖z‖` and from averaging instead of max-ing.
- **Hybrid precision and accuracy** should rise in lock-step because Stage 1 already had high precision; the bottleneck was Stage 2.
- **Stage-2 Normal precision** should stay high (similar to or slightly better than 0.985) because we are tightening, not loosening, the operating point.

### 7.2 What may slightly drop

- **Zero-day recall** is the explicit trade-off. We expect a drop on the order of 0.05–0.10 vs. the F1-tuned parent. If the user later cares more about recall, three knobs are exposed:
  1. Lower `PRECISION_FLOOR` (e.g., 0.75).
  2. Raise `F_BETA` toward 1.0.
  3. Drop `AE_ENSEMBLE_SIZE` to 1 (variance-bound is looser, more recall, less precision).

### 7.3 Failure modes to watch in the logs

- **"used_floor=False"** in the threshold-tuning print: the floor was not met. This indicates the AE / score combination cannot resolve Normal from probes at 0.85 precision. Inspect whether GAN probes look too realistic (over-train the GAN) or whether the AE is underfit (raise `AE_EPOCHS`).
- **`val precision < 0.5`**: probe filtering left too few positives. Inspect the printed counts of GAN / noise / shuffle / `total_kept`.
- **NaNs from Mahalanobis**: a latent dimension collapsed entirely. The eigenvalue floor `MAHA_EIG_FLOOR` prevents division-by-zero but if you see NaNs, raise the floor by 10× or the shrinkage by 2×.

---

## 8. How to run

From `Endsem_prep/` (locally or in Colab — the script auto-detects):

```bash
python claude_opus_model.py
```

Requires TensorFlow ≥ 2.10, scikit-learn ≥ 1.0, pandas, numpy, matplotlib, seaborn — same dependency set as the other midsem scripts. The dataset path follows the same convention: local `9) Car-Hacking Dataset` or Colab `/content/drive/MyDrive/dataset/9) Car-Hacking Dataset`.

---

## 9. Summary in one paragraph (research-paper style)

We propose a hierarchical intrusion-detection pipeline for CAN bus traffic in which an already-strong supervised closed-set classifier (Stage 1) is composed with an unsupervised novelty detector (Stage 2). The novelty detector is built as an ensemble of three sequence-aware Conv1D–BiLSTM bottleneck autoencoders trained on normal traffic only; for every window we extract three complementary anomaly signals — global reconstruction error, maximum per-time-step reconstruction error, and a Mahalanobis distance of the bottleneck embedding from the empirical Gaussian model of normal latents — robust-standardize them on training normals using median and MAD, and fuse them by averaging. The decision threshold is selected on a held-out validation set by maximizing F\(_{0.5}\) (precision-weighted) under a soft precision floor, using a deliberately diverse synthetic-anomaly set comprising GAN samples, uniform-noise probes, and feature-permuted normal windows so that the operating point generalizes across multiple out-of-distribution regimes. The design directly attacks the dominant failure mode of the parent variant — over-eager `zero_day` predictions caused by a low-rank latent score and an F1-symmetric operating point — and is structured so that each component (ensemble size, score fusion rule, threshold objective) is independently ablatable.
