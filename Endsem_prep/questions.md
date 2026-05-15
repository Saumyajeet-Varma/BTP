# Presentation Q&A — claude_opus_model.py

> Answers are written for a presentation audience — clear, direct, and backed by the actual design decisions in the pipeline.

---

## Q1. Explain the pipeline

The pipeline is a **two-stage hybrid Intrusion Detection System (IDS)** for CAN bus traffic in connected vehicles.

**What is CAN bus?**
The Controller Area Network (CAN) bus is the internal communication backbone of a vehicle — ECUs (Engine Control Units) broadcast short messages (frames) on it constantly. An attacker who gains access to this bus can inject malicious frames to manipulate steering, brakes, or the engine.

**The core problem**
Known attacks (DoS, Fuzzy, Gear spoofing, RPM spoofing) can be detected with a trained classifier. But the real threat is an **unknown or zero-day attack** — one that was never seen during training. A pure classifier cannot handle this, so we need a two-stage design.

---

### Stage 1 — Closed-Set Supervised Classifier

- Reads a **sliding window of 24 consecutive CAN frames**, each described by 14 features: CAN ID, DLC, 8 data bytes, and 4 engineered features (inter-arrival time, CAN ID frequency, byte entropy, byte statistics).
- Predicts one of **5 classes**: Normal, DoS, Fuzzy, Gear, RPM.
- If Stage 1 predicts any known attack → that is the final label. The window does not proceed further.
- If Stage 1 predicts **Normal** → the window is passed to Stage 2 for deeper inspection.

**Why Stage 1 alone is not enough:**
Stage 1 is a supervised model — it can only classify what it was trained on. A zero-day attack it has never seen will look "Normal" to it, so it passes through. Stage 2 is designed to catch exactly those windows.

---

### Stage 2 — Open-Set Anomaly Detector

- Receives only windows Stage 1 labelled as Normal.
- Computes an **anomaly score** using an ensemble of autoencoders trained exclusively on normal traffic.
- Compares the score against a tuned threshold `T`:
  - Score ≤ T → **Normal** (confirmed)
  - Score > T → **zero_day** (flagged as unknown attack)

The full decision rule is:

```
If Stage 1(x) ≠ Normal  →  output that known attack label
If Stage 1(x)  = Normal  and  score(x) > T  →  zero_day
If Stage 1(x)  = Normal  and  score(x) ≤ T  →  Normal
```

---

### Data Flow Summary

```
Raw CAN stream
      ↓
  Parse frames → sort by timestamp → engineer features
      ↓
  Sliding window (size 24) → scale to [0, 1]
      ↓
  Stratified split: 80% train_big | 20% test
      ↓
  train_big further split: ~68% X_fit | ~12% X_val
      ↓
  X_fit → train Stage 1 + train AE ensemble (on Normal only) + train GAN
  X_val → tune Stage-2 threshold (never touches AE weights)
  X_test → final evaluation
```

---

## Q2. Why this model? Explain all models used in each stage and their role

### Stage 1 Model — CNN + LSTM

**Architecture:**
```
Input (24, 14)
  → Conv1D(64) → BatchNorm → MaxPool → Dropout
  → Conv1D(128) → BatchNorm → MaxPool → Dropout
  → LSTM(96, return_sequences=True) → Dropout
  → LSTM(64) → BatchNorm → Dropout
  → Dense(128) → Dense(5, softmax)
```

**Why CNN first?**
CAN frames within a window have local temporal patterns — a DoS attack floods the bus with repeated IDs in short bursts. Conv1D with a kernel of size 3 picks up these short-range motifs across consecutive frames efficiently without needing to learn position-by-position independently.

**Why LSTM after CNN?**
After local pattern extraction, LSTM captures the **sequence-level context** — how the pattern evolves over the full 24-frame window. A DoS burst looks different from normal traffic not just frame-by-frame but in how it sustains over time. LSTM remembers that.

**Why class weights?**
The dataset is imbalanced — normal traffic appears far more often than any single attack type. Class-weighted cross-entropy loss ensures the model does not just learn to predict "Normal" for everything. Each class weight = (total samples) / (num_classes × samples_in_that_class).

---

### Stage 2 Model — Ensemble of 3 Sequence-Aware Autoencoders

**Why an Autoencoder for anomaly detection?**
An autoencoder is trained to **compress and reconstruct** its training data. If you train it only on normal traffic, it learns the structure of normal CAN patterns. When it sees an anomalous window, it fails to reconstruct it well — that high reconstruction error is the anomaly signal.

**Architecture of each AE:**
```
Encoder:
  Input (24, 14)
  → Conv1D(64) → BatchNorm
  → Conv1D(96) → BatchNorm
  → Bidirectional LSTM(64)
  → Dropout → Dense(32, tanh)   ← bottleneck z

Decoder:
  Dense(24×32) → Reshape(24, 32)
  → LSTM(64) → BatchNorm
  → Conv1DTranspose(96) → Conv1DTranspose(64)
  → TimeDistributed(Dense(14, sigmoid))
```

**Why sequence-aware (not flat MLP)?**
The previous model flattened the (24, 14) window into a 336-D vector and fed it to a plain MLP. This destroyed the temporal structure — the fact that frame `t` and frame `t+1` are correlated. The Conv1D + BiLSTM encoder learns **temporal dependencies** natively, so normal traffic reconstructs with lower variance. A tighter Normal score distribution means a cleaner separation from anomalous traffic.

**Why Bidirectional LSTM in the encoder?**
A forward LSTM sees each frame conditioned on what came before. A bidirectional one also sees what comes after. In a 24-frame window, both past and future context help build a better compressed representation of the normal pattern.

**Why `tanh` in the bottleneck?**
`tanh` bounds the latent vector to `[-1, 1]`. This makes the latent space bounded and symmetric — a necessary property for **Mahalanobis distance** to be numerically stable. A ReLU bottleneck can produce heavy-tailed, zero-saturated latent vectors that break the Gaussian assumption underlying Mahalanobis.

**Why an ensemble of 3 AEs?**
A single AE has a noisy anomaly score — it depends on random initialization and the specific SGD trajectory. Two normal windows can get very different scores simply due to AE training variance. By training 3 independent AEs with different seeds and averaging their scores, we reduce this variance by ≈ √3. A lower-variance Normal score distribution means we can set a higher, tighter threshold — and that is the direct lever for improving zero-day precision.

---

### GAN — Probe Generator for Threshold Tuning

**Architecture:** Standard unconditional GAN (Generator + Discriminator), trained on flattened normal windows.

**Role:** The GAN is NOT used to train the AE. It is used only to generate **synthetic out-of-distribution (OOD) probe windows** for tuning the Stage-2 threshold. The generator learns to produce windows that look like normal CAN traffic — these are then used as stand-ins for "what a zero-day might look like to Stage 2."

**Why GAN-generated samples as zero-day proxies?**
At training time, we have no real zero-day samples. The GAN generates plausible-but-not-real windows drawn from the vicinity of the normal manifold — exactly where a stealthy zero-day attack would sit. This gives Stage-2 threshold tuning a realistic evaluation surface.

---

## Q3. Explain the training and testing flow

### Training Flow

```
Step 1 — Data preparation
  Load normal_run_data.txt + 4 attack CSVs
  Parse → sort by timestamp → add engineered features
  Sliding window of size 24 → MinMaxScaler to [0,1]
  Stratified split:  80% train_big / 20% test
  Inner split of train_big:  ~68% X_fit / ~12% X_val

Step 2 — Stage 1 training (on X_fit, all classes)
  CNN-LSTM classifier, 5 output classes
  Class-weighted categorical cross-entropy
  EarlyStopping + ReduceLROnPlateau
  Trained on X_fit, validated on 15% of X_fit internally

Step 3 — Stage 2 AE ensemble training (on Normal-only rows of X_fit)
  Filter X_fit to Normal windows only  →  X_fit_normal
  Train 3 independent Conv1D+BiLSTM AEs with different seeds
  Each AE minimizes MSE(input, reconstruction) on X_fit_normal
  EarlyStopping on 10% internal val split of X_fit_normal
  After each AE trains:
    → Encode X_fit_normal → get Z_normal
    → Fit Mahalanobis params (mean + shrunk precision matrix) on Z_normal
    → Store robust standardization stats (median, MAD) on 3 raw scores

Step 4 — GAN training (on flat Normal windows of X_fit)
  Train discriminator on real normals vs. generator fakes
  Train generator to fool discriminator
  30 epochs × 130 steps — produces a probe generator

Step 5 — Stage-2 threshold tuning (on X_val, never seen by AEs)
  Build NEGATIVES:  Normal windows in X_val where Stage 1 also predicts Normal
  Build POSITIVES (diverse OOD):
    → 2500 GAN-generated windows
    → 1500 uniform noise windows
    → 1500 feature-shuffled normal windows
    Keep only those Stage 1 predicts as Normal (they reach Stage 2 in deployment)
  Score all neg + pos windows with the ensemble scorer
  Grid search 600 threshold candidates between 1st and 99.5th percentile
  Pick T that maximizes F0.5 (precision-weighted) with precision ≥ 0.85 floor
```

### Testing Flow

```
Step 6 — Evaluation on X_test + GAN test windows
  Generate 1500 fresh GAN windows → labelled zero_day
  Combine with X_test (real, 5 classes)

  For each window:
    Run Stage 1
    If Stage 1 ≠ Normal → final label = that known attack
    If Stage 1 = Normal:
      Compute ensemble anomaly score
      If score > T → final label = zero_day
      If score ≤ T → final label = Normal

  Report three metric blocks:
    (a) Stage 1 only — 5-class metrics
    (b) Stage 2 only — binary Normal vs zero_day, restricted to Stage1=Normal windows
    (c) Final hybrid — 6-class metrics (Normal, DoS, Fuzzy, Gear, RPM, zero_day)
```

### Key principle: no data leakage

| Data split | Used for |
|------------|----------|
| `X_fit` | Train Stage 1 + Train AE ensemble + Train GAN |
| `X_val` | Tune threshold T only (AE weights never updated on this) |
| `X_test` | Final evaluation only — never touched during training or tuning |

---

## Q4. What is a zero_day attack here?

### Real-world definition
A **zero-day attack** is an attack that exploits a vulnerability or uses a technique that was **unknown at the time the defense was built**. The defender has had zero days to prepare for it.

### What it means in this pipeline
In this system, **zero_day is the label for any CAN bus attack pattern the Stage-1 classifier was never trained on.**

Stage 1 is trained on four specific labeled attack families: DoS, Fuzzy, Gear spoofing, RPM spoofing. Any attack that is structurally different from these — a novel payload injection, a timing-based side-channel, a new spoofing variant — would look like Normal traffic to Stage 1 and pass through undetected.

### How zero_day traffic is simulated in this work
Since real unknown attacks are, by definition, unavailable during development, we use a **GAN (Generative Adversarial Network)** trained on real normal traffic to synthesize **plausible-but-novel** windows as proxies for zero-day traffic.

The GAN generator learns the statistical distribution of normal CAN traffic and produces new samples that:
- Are statistically similar to normals (so Stage 1 cannot distinguish them)
- Are not identical to any real normal window (they are off the true normal manifold)
- Represent the regime where a stealthy unknown attack would live

These GAN-generated windows are labelled `zero_day` for evaluation. At test time, 1500 fresh GAN windows are mixed into the test set and the system is evaluated on its ability to flag them.

### Why this is a meaningful stand-in
A stealthy attacker would craft traffic that looks normal to known classifiers but deviates subtly from genuine normal traffic. That is exactly what the GAN produces — it is trained to fool a discriminator that has seen real normals. The Stage-2 anomaly detector must find the deviation in the reconstruction residual or the latent space geometry, not in the label space.

### What Stage 2 does with it
Stage 2 never sees the zero-day label during training. It is trained purely on normals. At inference, it scores every Stage-1-Normal window and flags it as `zero_day` if the anomaly score exceeds threshold `T`. The quality of this detection is measured by precision and recall on the `zero_day` class.

**Previous model:** zero_day precision = 0.53 (roughly a coin flip — almost half the windows it called zero_day were actually Normal).
**This model's target:** zero_day precision ≥ 0.85 (using the precision-floor constraint in threshold tuning).

---

## Q5. Explain Mahalanobis distance, and why choose it over z-score

### What is z-score (the old approach)?

A z-score measures how far a single number is from a mean in units of standard deviation:

```
z = |x - μ| / σ
```

The previous model computed the z-score of the **L2 norm of the latent vector** `‖z‖`:

```
z_lat = |‖z‖ - μ_‖z‖| / σ_‖z‖
```

This collapses the entire 32-dimensional latent vector into **one number** (its length) before measuring how unusual it is. All directional information is thrown away.

---

### What is Mahalanobis distance?

Mahalanobis distance measures how far a point is from the **center of a distribution**, taking into account both the **scale** and the **correlations** of all dimensions together.

**Formula:**

```
d_M(z) = sqrt( (z - μ)ᵀ · Σ⁻¹ · (z - μ) )
```

Where:
- `z` is the latent vector (32-D in this model)
- `μ` is the mean latent vector of all train-Normal windows
- `Σ⁻¹` is the inverse of the covariance matrix of train-Normal latents

**Intuition:** Imagine normal latents form an ellipse in 2D (an ellipsoid in 32D). Euclidean distance treats all directions equally. Mahalanobis distance **stretches the space** so that directions of high variance (where normals naturally spread out) are downweighted, and directions of low variance (where normals are tightly clustered) are upweighted. A point at the edge of the ellipsoid — which is still inside the normal distribution — gets a moderate distance. A point that's just a short Euclidean step away but in an unusual direction gets a large Mahalanobis distance.

---

### Why Mahalanobis over z-score on ‖z‖?

| Property | z-score on ‖z‖ | Mahalanobis on z |
|----------|----------------|------------------|
| Dimensions used | 1 (the norm) | All 32 |
| Captures correlations between latent dims | No | Yes |
| Scale-invariant per dimension | No | Yes |
| Catches directional anomalies | No | Yes |
| Catches magnitude anomalies | Yes | Yes |

**Concrete example:** Suppose a zero-day attack shifts latent dimensions 3 and 7 in opposite directions. The L2 norm stays roughly the same (the shifts cancel). The z-score on `‖z‖` sees nothing. Mahalanobis sees the point is off the normal ellipsoid in the (dim3, dim7) plane and correctly gives it a high distance.

**Another way to say it:** z-score on `‖z‖` only checks *how far from the origin* the latent is. Mahalanobis checks *how far from the normal population manifold* the latent is. For anomaly detection, the second question is always the right one.

---

### Numerical safeguards in this implementation

Inverting a 32×32 covariance matrix estimated from ~6,000–10,000 samples can be ill-conditioned — some latent dimensions may be nearly constant, giving near-zero eigenvalues and near-infinite precision values.

Two safeguards are applied:

1. **Ledoit-Wolf style shrinkage (λ = 0.05):**
   ```
   Σ_shrunk = (1 - λ) · Σ  +  λ · diag(Σ)
   ```
   This blends the empirical covariance with a purely diagonal version. It moves extreme off-diagonal covariances slightly toward zero — reducing the effect of estimation noise in low-sample-count directions.

2. **Eigenvalue floor (ε = 1e-4):**
   After eigen-decomposition, any eigenvalue below `1e-4` is clipped to `1e-4` before inverting. This prevents any latent direction from blowing up the precision matrix due to near-zero variance.

Together these make the Mahalanobis computation **numerically stable** across different AE seeds and dataset samples without distorting the dominant structure of the normal latent distribution.

---

## Q6. What is different in our `claude_opus_model.py` from the CANGuard base paper (`2603.25763v1.pdf`)?

**Base paper:** *CANGuard: A Spatio-Temporal CNN-GRU-Attention Hybrid Architecture for Intrusion Detection in In-Vehicle CAN Networks* (arXiv:2603.25763v1).

**Our work:** `claude_opus_model.py` — a **two-stage hybrid IDS** inspired by CANGuard’s supervised backbone but extended to handle **unknown (zero-day) attacks**, which the paper explicitly leaves as future work.

---

### One-line summary (for slides)

> **CANGuard** = one supervised model, closed-set, known attacks only.  
> **Our pipeline** = CANGuard-style Stage 1 + a new unsupervised Stage 2 for zero-day detection on traffic Stage 1 calls Normal.

---

### Side-by-side comparison

| Aspect | CANGuard (base paper) | Our `claude_opus_model.py` |
|--------|----------------------|----------------------------|
| **Overall design** | Single-stage, end-to-end classifier | **Two-stage hybrid** (supervised + unsupervised) |
| **Problem scope** | **Closed-set** — only classes seen at training time | **Open-set** — adds a 6th label: **zero_day** |
| **Stage 1 backbone** | CNN → stacked **BiGRU** → **attention** → FC → softmax | CNN → **LSTM** (no attention block) → FC → softmax |
| **Stage 2** | **None** | Ensemble of **3 sequence-aware autoencoders** + Mahalanobis + multi-score fusion |
| **Unknown attacks** | Not detected (must be one of 6 trained classes) | Flagged as **zero_day** when anomaly score > threshold |
| **Dataset** | **CICIoV2024** (~1.4M samples, 12 features) | **Car-Hacking Dataset** (DoS, Fuzzy, Gear, RPM + Normal) |
| **Classes** | BENIGN, DoS, GAS, RPM, SPEED, STEERING WHEEL | Normal, DoS, Fuzzy, Gear, RPM, **zero_day** |
| **Normalization** | **Z-score** per feature | **MinMaxScaler** to [0, 1] |
| **Imbalance handling** | **BorderlineSMOTE** on flattened training windows | **Class weights** in Stage-1 loss (no SMOTE) |
| **Stage-2 training data** | N/A | **Normal traffic only** (AE never sees attacks) |
| **Zero-day evaluation** | Not in paper | **GAN + noise + feature-shuffle** probes; threshold tuned on validation |
| **Anomaly scoring** | N/A | Mean MSE + max-step MSE + **Mahalanobis** in latent; robust median/MAD; **ensemble average** |
| **Threshold / operating point** | Argmax softmax | Grid search on validation; **F0.5** (precision-priority) + precision floor ≥ 0.85 |
| **Interpretability** | **SHAP** on payload bytes | Confusion matrices (Stage 1, Stage 2, hybrid) — no SHAP in this script |
| **Framework** | Not specified in summary (typical PyTorch/TF in such papers) | **TensorFlow / Keras** |
| **Regularization** | Dropout 0.3, L2 (λ=0.001), grad clipping | Dropout in CNN/LSTM/AE; no L2 in this script; EarlyStopping + ReduceLROnPlateau |

---

### What we kept (in spirit of CANGuard)

1. **Spatio-temporal idea for known attacks** — Stage 1 still uses **1D convolutions** for local frame patterns and a **recurrent layer** (LSTM instead of BiGRU) for sequence context. Same high-level “CNN + temporal model” philosophy as the paper.

2. **Sliding windows** — Both pipelines segment the CAN stream into fixed-length windows and label by the window (label at end of window in CANGuard; same idea in our script).

3. **Multi-class supervised IDS on CAN** — Stage 1 is the direct analogue of CANGuard’s role: classify **known** attack types vs Normal.

4. **Class imbalance awareness** — CANGuard uses SMOTE + class weights; we use **balanced class weights** on Stage 1 (different technique, same goal).

---

### What we added (our contribution beyond the paper)

1. **Second stage for open-set / zero-day detection**  
   CANGuard’s own limitations section states: no handling of attacks outside the training label set. We implement exactly what their “future work” suggests: a model trained **only on Normal** that screens windows Stage 1 still calls Normal.

2. **Hybrid decision rule**  
   ```
   If Stage 1 ≠ Normal  →  known attack label
   If Stage 1 = Normal and score > T  →  zero_day
   If Stage 1 = Normal and score ≤ T  →  Normal
   ```
   CANGuard has no such branch — every sample gets a single softmax class.

3. **Sequence-aware autoencoder ensemble (Stage 2)**  
   - Conv1D + BiLSTM bottleneck AE on `(24, 14)` windows (not a flat MLP).  
   - **3 AEs** with different seeds; scores averaged for stability.  
   - **Mahalanobis distance** in 32-D latent space (with covariance shrinkage), not just reconstruction error.

4. **Multi-signal anomaly score**  
   Three complementary signals fused with robust standardization (median/MAD) and averaged: global reconstruction error, **max per-timestep** reconstruction error (localized spoof), and latent Mahalanobis distance.

5. **GAN + diverse OOD probes for calibration**  
   CANGuard does not use generative models. We train a GAN on normal flats to synthesize plausible unknowns, plus uniform-noise and feature-shuffled normals, to tune the Stage-2 threshold without real zero-day labels.

6. **Precision-prioritized threshold (F0.5 + precision floor)**  
   Tuned on a held-out validation set that Stage 2 never trained on — targets high **zero_day precision** under Normal-heavy traffic at test time.

---

### What we simplified or changed vs CANGuard

| CANGuard feature | Our choice | Why |
|------------------|------------|-----|
| **Attention** after BiGRU | Not used in Stage 1 | Simpler backbone; LSTM + dropout already strong on Car-Hacking subset; compute saved for Stage 2 ensemble |
| **BiGRU** | **LSTM** (uni-directional stack) | Same temporal modeling family; fewer parameters; aligned with earlier midsem scripts |
| **SHAP interpretability** | Not in this file | Focus of script is hybrid + zero-day metrics, not byte-level explanation |
| **BorderlineSMOTE** | Not used | Avoids synthetic **labeled** attack windows that could blur Stage-1 boundaries; Stage 2 uses separate **unlabeled** synthetic probes instead |
| **Z-score normalization** | **MinMax** to [0, 1] | Matches AE/GAN sigmoid outputs and flat probe space |

---

### Dataset difference (important for viva)

- **CANGuard** is evaluated on **CICIoV2024** (IoV benchmark, different attack names and scale).  
- **We** use the **Car-Hacking Dataset** (academic CAN intrusion benchmark with DoS, Fuzzy, Gear, RPM).  

So we are **not replicating CANGuard’s exact experiment** — we are **adapting its supervised IDS idea** to a different dataset and **extending the architecture** with Stage 2. Fair comparison sentence: *“Stage 1 follows the spatio-temporal supervised IDS paradigm of CANGuard; Stage 2 and zero_day handling are our extensions.”*

---

### Presentation slide bullets

- Base paper: **single-model, closed-set**, CICIoV2024, CNN–BiGRU–Attention.  
- Ours: **two-stage**, Car-Hacking, Stage 1 = CNN–LSTM; Stage 2 = **AE ensemble + Mahalanobis + GAN-calibrated threshold**.  
- Main gap filled: **zero-day / unknown attacks** on Normal-looking traffic.  
- CANGuard future work → **implemented** as unsupervised second stage on Normal-only training data.

---

## Q7. Explain the dataset, data preprocessing, and feature engineering

*(For `claude_opus_model.py` — Car-Hacking Dataset pipeline.)*

---

### 7.1 The dataset — Car-Hacking Dataset

**What it is:** A public **in-vehicle CAN bus intrusion detection** benchmark. Real CAN traffic was captured from a vehicle (or test setup) under **normal driving** and under **injected attacks** on the bus.

**Why we use it:** It provides labelled known attacks (for Stage 1) and long normal traces (for Stage 2 AE training and GAN probes). It is widely used in CAN IDS research and matches our attack taxonomy (DoS + spoofing-style attacks).

**Files used in the script:**

| File | Format | Role |
|------|--------|------|
| `normal_run_data.txt` | Text log (regex-parsed) | Benign CAN traffic only |
| `DoS_dataset.csv` (or `dos_attack.csv`) | CSV | DoS attack + mixed R/T flags |
| `Fuzzy_dataset.csv` (or `fuzzy_attack.csv`) | CSV | Fuzzy attack |
| `gear_dataset.csv` (or `gear_spoofing.csv`) | CSV | Gear spoofing |
| `RPM_dataset.csv` (or `rpm_spoofing.csv`) | CSV | RPM spoofing |

**Subset mode (default):** For faster experiments, `USE_SUBSET = True` caps:
- **10,000** normal frames from the text log
- **20,000** rows per attack CSV

Set `USE_SUBSET = False` to use the full dataset.

**Classes after preprocessing:**

| Label | Source |
|-------|--------|
| Normal | Normal log, or attack CSV rows with Flag = **R** (recovery / benign segment) |
| DoS, Fuzzy, Gear, RPM | Attack CSV rows with Flag = **T** (attack segment) |
| zero_day | **Not in raw data** — synthetic GAN windows at test/eval time only |

Stage 1 is trained on **5 classes** (no zero_day in training labels). zero_day appears only in evaluation.

---

### 7.2 Raw CAN frame — what each row contains

Every frame (one row before windowing) has:

| Field | Meaning |
|-------|---------|
| **Timestamp** | Time of the frame (seconds) |
| **CAN_ID** | Identifier of the sending ECU / message type (hex in raw files) |
| **DLC** | Data Length Code — how many payload bytes are valid (0–8) |
| **DATA0 … DATA7** | Up to 8 payload bytes (hex values 0–255) |

**Example (normal text line):**
```
Timestamp: 1479121434.850202  ID: 0350  000  DLC: 8  05 28 84 66 6d 00 00 a2
```

The script parses this with a regex into structured fields.

---

### 7.3 Data loading — two different parsers

#### A) Normal traffic — `load_normal_df`

1. Opens `normal_run_data.txt` (checks root or `normal_run_data/` subfolder).
2. Reads line by line; **`parse_line`** extracts Timestamp, CAN_ID (hex → int), DLC, and 8 data bytes.
3. Pads data to 8 bytes if shorter; splits into `DATA0` … `DATA7`.
4. Sets **`Label = "Normal"`** for every row.

#### B) Attack traffic — `load_attack_df`

1. Reads CSV with columns: `Timestamp, CAN_ID, DLC, DATA0–DATA7, Flag`.
2. **`convert_numeric_columns`:** CAN_ID and DATA bytes may be hex strings → converted to integers; invalid values → 0.
3. **`label_from_flag`:**
   - Flag **`R`** → **Normal** (benign segment inside an attack recording)
   - Flag **`T`** → **attack label** (DoS / Fuzzy / Gear / RPM)
   - Anything else → Normal (safe default)

**Why Flag matters:** Attack files contain *both* attack and recovery/normal periods. Using Flag avoids labelling entire files as attack when half the rows are benign.

---

### 7.4 Feature engineering — `add_engineered_features`

Applied **per stream** (each file sorted by Timestamp first). We start with **10 base features** and add **6 engineered features** → **14 features per frame**.

#### Base features (10)

```
CAN_ID, DLC, DATA0, DATA1, DATA2, DATA3, DATA4, DATA5, DATA6, DATA7
```

These are the raw CAN semantics the model sees directly.

#### Engineered features (6) — why each exists

| Feature | Formula / idea | Why it helps IDS |
|---------|----------------|------------------|
| **IAT** (Inter-Arrival Time) | `diff(Timestamp)`, first row = 0, clipped to [0, 1] | DoS floods the bus → **very small IAT**; spoofing may change timing patterns |
| **CAN_ID_freq** | Global frequency of each CAN_ID in that file (normalized count) | Rare IDs during an attack stand out; common IDs during normal driving get high freq |
| **byte_entropy** | Shannon entropy of non-zero DATA bytes in the row | Random/fuzzy payloads → **high entropy**; stable normal payloads → lower entropy |
| **byte_sum** | Sum of DATA0–DATA7 | Captures overall payload magnitude shift under spoofing |
| **byte_range** | max(DATA) − min(DATA) | Spread of byte values — attacks often widen or narrow this |
| **byte_std** | Standard deviation across DATA bytes | Variability within one frame’s payload |

**Entropy detail:** Only non-zero bytes are used; if all zero, entropy = 0. Uses log₂ with a small epsilon for numerical stability.

**IAT clipping to 1 second:** Prevents one huge timestamp gap from dominating; keeps scale reasonable before MinMax scaling.

---

### 7.5 Sliding windows — `make_windows_from_sorted_df`

**Window size:** `SEQ_LEN = 24` consecutive frames.

**Input tensor per window:** shape `(24, 14)` — 24 time steps, 14 features each.

**Label rule:** The window label is the **label of the last frame** in the window (`y_str[i + seq_len - 1]`).

```
Frames:  [f0, f1, f2, ..., f22, f23]
                              ↑
                         window label
```

**Why sliding windows?**
- Stage 1 (CNN-LSTM) and Stage 2 (sequence AE) need **temporal context** — one frame alone is weak; 24 frames capture bursts, timing, and short patterns.
- Standard practice in CAN IDS (same idea as CANGuard’s windowing, different length).

**Building the full dataset:** `build_all_windows` loads normal + 4 attack files separately, windows each stream, then **concatenates** all windows into one array `X_w` and label array `y_str`.

---

### 7.6 Preprocessing after windowing

#### Step 1 — Label encoding

String labels → integers 0–4 for Stage 1 (`LabelEncoder` on Normal, DoS, Fuzzy, Gear, RPM).

#### Step 2 — MinMax scaling

```text
Flatten each window: (24, 14) → 336-D vector
MinMaxScaler.fit_transform on all windows → values in [0, 1]
Reshape back to (24, 14)
```

**Why MinMax to [0, 1]?**
- Neural nets train more stably on bounded inputs.
- Stage 2 decoder uses **sigmoid** → outputs [0, 1]; inputs in the same range match.
- GAN generator also outputs sigmoid in [0, 1] for flat probes.

**Note:** Scaler is fit on **all windows together** before the train/test split (feature-range calibration only; labels are not used in scaling).

#### Step 3 — Stratified splits

```text
100% windows
    ├── 80% train_big  (stratified by class)
    │       ├── ~68% X_fit   → train Stage 1, AE, GAN
    │       └── ~12% X_val   → tune Stage-2 threshold only
    └── 20% X_test         → final evaluation (never used in training/tuning)
```

**Why stratified?** Keeps similar class proportions in fit, val, and test — important because Normal dominates.

#### Step 4 — Stage-2-specific subset

From `X_fit`, only rows with label **Normal** → `X_fit_normal_seq` / `X_fit_normal_flat`.

- **AE ensemble** trains on these only.
- **GAN** trains on flattened normal windows only.

Attacks are **not** shown to the AE during training — that is what makes Stage 2 an anomaly detector.

---

### 7.7 End-to-end preprocessing flow (diagram)

```text
Raw files (txt + CSV)
        ↓
   Parse & type-convert (hex → int, Flag → label)
        ↓
   Sort by Timestamp per file
        ↓
   Engineer 6 features (IAT, freq, entropy, sum, range, std)
        ↓
   Sliding window (length 24), label = last frame
        ↓
   Concatenate all classes → X_w (N, 24, 14), y_str
        ↓
   LabelEncoder + MinMaxScaler [0,1]
        ↓
   Stratified split → X_fit / X_val / X_test
        ↓
   X_fit Normal-only → AE + GAN training
```

---

### 7.8 Presentation talking points

1. **Dataset:** Car-Hacking — real CAN logs, 4 known attacks + normal; Flag column separates attack vs benign rows inside attack files.

2. **14 features per frame:** 10 raw CAN fields + 6 engineered (timing, ID frequency, payload statistics).

3. **Windows of 24 frames** give temporal context for CNN-LSTM and sequence AE.

4. **MinMax [0,1]** aligns with sigmoid AE/GAN and stabilizes training.

5. **Stratified splits** preserve class balance; **AE sees only Normal** — core of zero-day detection design.

6. **zero_day** is not in the raw dataset — it is evaluated using **synthetic GAN windows** at test time.

---

## Q8. Give a summary of the base paper (`2603.25763v1.pdf`)

**Paper:** *CANGuard: A Spatio-Temporal CNN-GRU-Attention Hybrid Architecture for Intrusion Detection in In-Vehicle CAN Networks*  
**Source:** arXiv:2603.25763v1 (2026)

### 1) Problem the paper addresses

Modern connected vehicles rely on the CAN bus, which does not provide built-in authentication/encryption. This makes CAN messages vulnerable to injection and spoofing attacks. The paper targets **intrusion detection** for this setting.

### 2) Core idea (model contribution)

CANGuard proposes a **single-stage supervised deep architecture**:

- **1D CNN** blocks for local/spatial feature extraction from CAN windows
- **Stacked bidirectional GRU** layers for temporal dependency learning
- **Attention mechanism** to focus on important time steps/features
- **Fully connected + softmax** for final multi-class prediction

In short: CNN captures local patterns, BiGRU captures sequence context, attention improves focus before classification.

### 3) Dataset and setup

- Evaluated on **CICIoV2024** (IoV CAN intrusion benchmark)
- Uses sliding windows and standard preprocessing
- Addresses class imbalance (paper summary mentions BorderlineSMOTE + class weighting)
- Trained with Adam, cross-entropy, early stopping, regularization

### 4) Main results (high-level)

- Strong classification metrics (accuracy / precision / recall / F1) on known attack classes
- Ablation study shows benefit of combining CNN + GRU + attention over simpler variants
- SHAP-based analysis gives feature-importance interpretation for payload bytes

### 5) Paper limitations (important for viva)

The paper itself is mainly a **closed-set** supervised IDS study:

- Evaluated offline on one benchmark
- No on-vehicle real-time deployment study
- No explicit adversarial-robustness study
- No dedicated open-set/unknown (`zero_day`) detection module

### 6) Why it is the base for our work

CANGuard gives a strong Stage-1 philosophy: supervised spatio-temporal modeling for known attacks.  
Our `claude_opus_model.py` keeps that spirit for known classes and extends it with a Stage-2 anomaly detector to handle unknown/zero-day behavior that closed-set models miss.

### 7) 20-second presentation version

> CANGuard is a supervised CNN-BiGRU-Attention CAN IDS for known attacks on CICIoV2024.  
> It performs strongly in closed-set classification but does not explicitly solve zero-day detection.  
> Our work builds on that idea and adds a second unsupervised stage to detect unknown attacks among traffic Stage 1 considers normal.

---

## Q9. How can we say our model is based on this base paper? (Explain how)

This is a common viva question: *“You say CANGuard is your base paper — in what sense is your work actually based on it?”*  
Below is an honest, defensible way to explain it.

---

### Short answer (use this first)

> **Our work is based on CANGuard in problem formulation and Stage-1 design philosophy — spatio-temporal deep learning for CAN intrusion detection — and we extend it with a second stage for zero-day detection, which the paper identifies as future work but does not implement.**

We are **not** claiming we copied CANGuard’s exact architecture, dataset, or results. We are claiming we **inherit its core supervised IDS idea** and **extend** it for open-set / unknown attacks.

---

### What “based on” means in research (three levels)

| Level | Meaning | Does our work qualify? |
|-------|---------|------------------------|
| **1. Problem & domain** | Same application: CAN bus IDS for connected vehicles | **Yes** — same threat model (DoS, spoofing, malicious frames on CAN) |
| **2. Methodology** | Same high-level approach: deep learning on windowed CAN features with spatial + temporal modeling | **Yes for Stage 1** — CNN + recurrent temporal model on sliding windows |
| **3. Exact replication** | Same model layers, same dataset, same hyperparameters, same paper results | **No** — different dataset (Car-Hacking), different Stage-1 details (LSTM not BiGRU+Attention), plus entire Stage 2 |

**Correct phrasing:** *“Inspired by and extended from CANGuard”* or *“Built on the CANGuard paradigm with a novel second stage”* — not *“We implemented CANGuard as-is.”*

---

### Line-by-line: how our pipeline maps to CANGuard

```text
CANGuard (paper)                    Our claude_opus_model.py
─────────────────────────────────────────────────────────────────
CAN intrusion detection      →      Same goal
Sliding windows on CAN data  →      SEQ_LEN = 24 windows
Multi-class known attacks    →      Stage 1: Normal, DoS, Fuzzy, Gear, RPM
CNN for local patterns       →      Stage 1: Conv1D(64) → Conv1D(128)
Temporal model (BiGRU)       →      Stage 1: LSTM(96) → LSTM(64)  [same role]
Attention + FC + softmax     →      Stage 1: Dense + softmax (no attention block)
Class imbalance handling     →      Balanced class weights (different technique)
Closed-set only              →      Stage 2 added: zero_day on Normal-looking traffic
Future work: AE on normals   →      Implemented: AE ensemble + Mahalanobis + GAN probes
```

**Stage 1 is the direct conceptual descendant of CANGuard.**  
**Stage 2 is our contribution** — explicitly aligned with the paper’s stated future direction (unsupervised screening on benign-looking traffic).

---

### Four concrete reasons we can call it “based on” CANGuard

#### 1) Same problem statement

Both systems answer: *“Is this CAN traffic benign or malicious?”*  
CANGuard answers it with **one supervised classifier** over a fixed set of classes.  
We answer it with **supervised known attacks (Stage 1) + unsupervised unknown detection (Stage 2)** — but the **domain, inputs (CAN frames), and windowing idea** come from the same IDS line of work that CANGuard represents.

#### 2) Same spatio-temporal modeling philosophy

CANGuard’s main insight: CAN attacks are not visible in a single frame — you need **spatial** (per-frame / local) and **temporal** (across frames) modeling.

Our Stage 1 follows that blueprint:

- **Spatial / local:** Conv1D layers extract patterns across features and short time steps (like CANGuard’s CNN).
- **Temporal:** LSTM layers model how the window evolves over 24 frames (same *role* as CANGuard’s BiGRU, different cell type).

For presentation: *“Stage 1 implements the CANGuard-style spatio-temporal backbone for known attack classification.”*

#### 3) Same preprocessing paradigm

Both pipelines:

1. Parse CAN logs into structured rows (ID, DLC, payload bytes, timestamp).
2. Build **sliding windows** with a label tied to the window.
3. Scale features for neural network training.
4. Handle **class imbalance** during supervised training.

We use MinMax + class weights instead of Z-score + SMOTE, but the **pipeline structure** is the same class of solution CANGuard uses.

#### 4) We implement what CANGuard calls “future work”

From `CANGuard_base_paper_summary.md` (and the paper’s limitations / future work):

> Pairing this supervised architecture with a **second stage trained only on normal traffic** (e.g. autoencoder) is a natural way to address **open-set or unknown** attack behavior.

That sentence is essentially our **Stage 2 design document**. Our work is “based on” CANGuard in the sense that:

- We **accept** its supervised Stage-1 role for known attacks.
- We **extend** it where the paper stops — zero-day / unknown detection.

---

### How to say it in a presentation (ready-made sentences)

**Slide 1 — Base paper:**
> “We take **CANGuard** (CNN–BiGRU–Attention) as our reference architecture for **known-attack detection** on in-vehicle CAN traffic.”

**Slide 2 — Our extension:**
> “CANGuard is **closed-set**: it cannot label attacks outside its training classes. We add **Stage 2**: an autoencoder ensemble trained only on Normal traffic, with Mahalanobis-based scoring, to flag **zero-day** windows that Stage 1 still calls Normal.”

**Slide 3 — Relationship (one line):**
> “**Stage 1 = CANGuard paradigm; Stage 2 = our contribution beyond the paper.**”

**If examiner asks “Did you implement CANGuard?”**
> “We implemented the **same architectural idea** (CNN + temporal network on windowed CAN features) for Stage 1, adapted to the Car-Hacking dataset and TensorFlow. We did **not** replicate every layer (no attention block, LSTM instead of BiGRU). Our **novel part** is the hybrid second stage, which the base paper suggests but does not build.”

---

### What we should NOT claim (stay honest)

| Do not say | Say instead |
|------------|-------------|
| “We implemented CANGuard.” | “We adapted the CANGuard **spatio-temporal supervised IDS approach** for Stage 1.” |
| “We improved CANGuard’s accuracy on CICIoV2024.” | “We evaluated on **Car-Hacking** with a **two-stage** system; direct accuracy comparison to the paper is not apples-to-apples.” |
| “Our Stage 1 is identical to CANGuard.” | “Stage 1 is **inspired by** CANGuard: CNN + temporal model, but LSTM backbone and no attention.” |
| “CANGuard detects zero-day.” | “CANGuard does **not**; we add zero-day via Stage 2.” |

---

### Visual for slides: “Based on” = inherit + extend

```text
                    CANGuard (base paper)
                    ┌─────────────────────┐
                    │  CNN + BiGRU + Attn │
                    │  → softmax (6 cls)  │
                    │  closed-set only    │
                    └──────────┬──────────┘
                               │ inherit idea
                               ▼
                    Our Stage 1 (adapted)
                    ┌─────────────────────┐
                    │  CNN + LSTM         │
                    │  → softmax (5 cls)  │
                    └──────────┬──────────┘
                               │ extend (paper's future work)
                               ▼
                    Our Stage 2 (new)
                    ┌─────────────────────┐
                    │  AE ensemble ×3     │
                    │  Mahalanobis score  │
                    │  → zero_day if > T  │
                    └─────────────────────┘
```

---

### One-paragraph viva answer (memorize this)

> Our model is based on CANGuard because we adopt the same fundamental approach to CAN intrusion detection: we convert CAN traffic into sliding windows, extract spatio-temporal features with a convolutional front end and a recurrent temporal model, and use a supervised classifier for known attack types. CANGuard stops at closed-set classification and notes that unknown attacks require additional mechanisms. We directly address that gap by adding Stage 2 — an unsupervised autoencoder ensemble with Mahalanobis anomaly scoring — on traffic that Stage 1 labels as Normal. So CANGuard is the foundation for known-attack detection; our contribution is the hybrid extension for zero-day detection, which aligns with the base paper’s own suggested future direction.
