# CAN Intrusion Detection: Full Solution vs. Base Paper

This document describes **our full solution** (two-stage hybrid: LSTM classifier + autoencoder) from `model2025_cursor.py`, and the **base paper** (adaptive autoencoder-based IDS) in terms of solution overview, dataset, preprocessing, training/model, and rationale for each approach.

**Base paper:** Kim, D.; Im, H.; Lee, S. “Adaptive Autoencoder-Based Intrusion Detection System with Single Threshold for CAN Networks.” *Sensors* 2025, 25, 4174.

---

## Part 1 — Our Solution (Full Pipeline: LSTM + Autoencoder)

### 1.1 Solution Overview

Our solution is a **two-stage hybrid** intrusion detection system for Controller Area Network (CAN) traffic:

1. **Stage 1 — LSTM classifier (supervised):** Classifies each CAN message into one of five **known** classes: **Normal**, **DoS**, **Fuzzy**, **Gear** (gear spoofing), or **RPM** (RPM spoofing).
2. **Stage 2 — Autoencoder (unsupervised):** Applied only when Stage 1 predicts **Normal**. The autoencoder is trained only on normal data; if the reconstruction error for that message exceeds a threshold, the message is relabeled as **Unknown Attack**; otherwise it stays **Normal**.

So the pipeline combines:
- **Known-attack classification** via the LSTM (supervised, multi-class).
- **Unknown-attack detection** via the autoencoder (unsupervised, anomaly detection on “suspected normal” messages).

- **Learning paradigm:** Hybrid — supervised (Stage 1) + unsupervised (Stage 2).
- **Input unit:** One CAN frame: CAN_ID, DLC, and 8 data bytes (10 features).
- **Output:** Either a known class (Normal, DoS, Fuzzy, Gear, RPM) or **Unknown Attack** when Stage 1 says Normal but Stage 2 flags high reconstruction error.
- **Use case:** Classify known attacks and flag potential unknown attacks that look normal to the classifier but anomalous to the autoencoder.

### 1.2 Dataset

- **Source:** Car-Hacking Dataset from the Hacking and Countermeasure Research Lab  
  **Link:** https://ocslab.hksecurity.net/Datasets/car-hacking-dataset  

- **Content:**
  - **Normal:** Parsed from `normal_run_data/normal_run_data.txt` (timestamp, CAN ID, DLC, 8-byte data).
  - **Attacks:** Loaded from CSV files:
    - `dos_attack.csv`
    - `fuzzy_attack.csv`
    - `gear_spoofing.csv`
    - `rpm_spoofing.csv`

- **Schema (per message):**
  - `Timestamp`, `CAN_ID`, `DLC`, `DATA0`–`DATA7` (8 data bytes); CSVs also have `Flag`; we add `Label` (Normal, DoS, Fuzzy, Gear, RPM).

- **Usage:** All five classes are concatenated into one DataFrame. We use **full frame fields** (CAN_ID, DLC, DATA0–7) for both the LSTM and the autoencoder — same dataset family as the base paper, but with more features than ID-only.

### 1.3 Dataset Preprocessing

1. **Parsing**
   - **Normal:** Regex parsing of the TXT file to extract Timestamp, CAN_ID (hex → int), DLC, and DATA; DATA is zero-padded to 8 bytes.
   - **Attacks:** CSV read with columns `Timestamp`, `CAN_ID`, `DLC`, `DATA0`–`DATA7`, `Flag`; a `Label` column is set per file (DoS, Fuzzy, Gear, RPM).

2. **Feature set**
   - **Features used for both Stage 1 and Stage 2:**  
     `['CAN_ID', 'DLC', 'DATA0', 'DATA1', 'DATA2', 'DATA3', 'DATA4', 'DATA5', 'DATA6', 'DATA7']`  
   - **10 features** per message (1 ID + 1 DLC + 8 data bytes).

3. **Train/test split**
   - **Stratified** 80/20 train–test split on the full labeled dataset (all five classes).
   - **Scaler:** `MinMaxScaler` fitted **only on the training set**; applied to train and test (no leakage).

4. **Reshaping for Stage 1 (LSTM)**
   - Each sample is shaped as `(1, N_FEATURES)` i.e. `(1, 10)`: one time step, 10 features, for the LSTM.

5. **Stage 2 (autoencoder) data**
   - Autoencoder is trained only on **training samples whose label is Normal** (same 10 features, already MinMax-scaled). No attack data is used for autoencoder training.

6. **Labels**
   - `LabelEncoder` for class names → integers; `to_categorical` for one-hot targets in Stage 1. For hybrid evaluation, predictions can be the original five classes or **Unknown Attack**.

7. **Class imbalance (Stage 1 only)**
   - `compute_class_weight('balanced', ...)` on training labels, passed to the LSTM `fit()` to weight loss by inverse class frequency.

### 1.4 Training and Model

**Stage 1 — LSTM classifier**

- **Architecture:**  
  Input `(batch, 1, 10)` → **LSTM(128,** `return_sequences=False`**)** → **BatchNormalization** → **Dropout(0.4)** → **Dense(64, relu)** → **Dropout(0.3)** → **Dense(32, relu)** → **Dense(5, softmax)**.
- **Training:** Adam, `categorical_crossentropy`, up to 30 epochs, batch size 64, 10% validation split, class weights, **EarlyStopping** on `val_loss` (patience 5, restore best weights).

**Stage 2 — Autoencoder**

- **Architecture:**  
  Input `(batch, 10)` → **Dense(64, relu)** → **BatchNormalization** → **Dense(32, relu)** → **Dense(16, relu)** → **Dense(32, relu)** → **Dense(64, relu)** → **Dense(10, linear)**.
- **Training:** Only on **normal** training samples; input = target (reconstruction). Adam, **MSE** loss, up to 30 epochs, batch size 64, 10% validation split, **EarlyStopping** on `val_loss` (patience 5, restore best weights).

**Threshold (Stage 2)**

- Reconstruction MSE is computed on all **normal training** samples.
- **Threshold:** `mean(train_loss) + 3 * std(train_loss)` (statistical rule; no KDE).
- At inference: if Stage 1 predicts Normal and reconstruction MSE > threshold → **Unknown Attack**; else **Normal**.

**Hybrid prediction (inference)**

1. Scale the sample if needed (using the same fitted `MinMaxScaler`).
2. Run Stage 1 (LSTM). If predicted class ≠ Normal → return that class (DoS, Fuzzy, Gear, RPM).
3. If predicted class = Normal → run Stage 2 (autoencoder). If MSE > threshold → return **Unknown Attack**; else return **Normal**.

**Reproducibility:** `RANDOM_STATE=42` for NumPy and TensorFlow; same seed for `train_test_split`.

### 1.5 Why We Use This Model

- **LSTM (Stage 1):** Uses full frame (ID, DLC, data) for strong **known-attack** classification; sequence-ready if we later use multiple time steps; class weights handle imbalance.
- **Autoencoder (Stage 2):** Trained only on normal traffic, so it learns the “shape” of normal messages; **unknown** attacks that slip past the LSTM as “Normal” can still produce high reconstruction error and be flagged as **Unknown Attack**.
- **Hybrid design:** Combines supervised performance on known attacks with unsupervised capability to flag anomalies, similar in spirit to the base paper’s goal (detect unseen attacks) while keeping explicit labels for known types.
- **Single threshold for Stage 2:** One scalar threshold (mean + 3×std on normal training loss) keeps the pipeline simple; no per-attack thresholds.
- **Shared features and scaler:** Same 10 features and scaling for both stages simplify the pipeline and deployment.

**Note:** “Unknown Attack” in our evaluation is assigned when the LSTM predicts Normal but the autoencoder’s reconstruction error is high. True unknown attack types would need to be present in the test set to measure this explicitly; the current test set has only the five known labels.

---

## Part 2 — Base Paper (Adaptive Autoencoder-Based IDS with Single Threshold)

**Reference:** Kim, D.; Im, H.; Lee, S. “Adaptive Autoencoder-Based Intrusion Detection System with Single Threshold for CAN Networks.” *Sensors* 2025, 25, 4174.  
**DOI:** https://doi.org/10.3390/s25134174  

### 2.1 Solution Overview (Base Paper)

The base paper proposes a **lightweight, unsupervised** IDS for CAN networks, designed for **real-time, on-device** deployment (including FPGA):

- **Learning paradigm:** Unsupervised. The model is trained **only on normal** CAN traffic (CAN IDs only).
- **Core idea:** An **autoencoder** learns to reconstruct sequences of **N** consecutive **CAN IDs** (each ID represented as 29 bits). Attack traffic yields **higher reconstruction error**; a **single threshold** (derived via **Gaussian KDE**) separates normal from attack for **all four attack types** (DoS, Fuzzy, Gear spoofing, RPM spoofing).
- **Main contributions:**
  1. Unsupervised autoencoder so that **unseen** attack types can still be flagged as anomalies.
  2. **Single threshold** for all attack types (unlike e.g. NovelADS, which uses per-attack thresholds).
  3. **Lightweight** design: few parameters and low FLOPs, validated on FPGA with reduced LUTs, flip-flops, and power vs. existing FPGA-based IDSs.

### 2.2 Dataset (Base Paper)

- **Source:** Vehicle hacking dataset from the **Hacking and Countermeasure Research Lab** (same dataset family as ours).  
  **Link:** https://ocslab.hksecurity.net/Datasets/car-hacking-dataset  

- **Collection:** Y-cable at the OBD-II port of a **Hyundai YF Sonata**.

- **Composition (paper’s Table 2):**

| Data Type     | Total Frames | Normal Frames | Attack Frames |
|---------------|-------------:|--------------:|--------------:|
| Normal        | 988,987      | 988,987       | —             |
| DoS attack    | 3,665,771    | 3,078,250     | 587,521       |
| Fuzzy attack  | 3,838,860    | 3,347,013     | 491,847       |
| Gear attack   | 4,443,142    | 3,845,890     | 597,252       |
| RPM attack    | 4,621,702    | 3,966,805     | 654,897       |

- **Attack descriptions:**
  - **DoS:** Inject CAN ID `0x000` every 0.3 ms.
  - **Fuzzy:** Random CAN IDs and data every 0.5 ms.
  - **Spoofing (Gear / RPM):** Inject gear- or RPM-related messages every 1 ms.

- **Usage in the paper:**
  - **Training:** Only **normal** frames (CAN IDs).
  - **Threshold/frame-count selection:** Two-thirds of each attack type’s data used to compute KDE and optimal threshold and N.
  - **Test:** Remaining one-third of attack data (and normal) for final evaluation; attack data are **not** used in training.

### 2.3 Dataset Preprocessing (Base Paper)

1. **Input used:** **CAN ID only** (no DLC or data payload).
2. **Bit representation:** CAN IDs are expanded to **29 bits** using **zero-padding** (compatible with CAN 2.0A/2.0B).
3. **Sequence construction:** Consecutive CAN IDs are grouped into **N frames** per sample, forming **2D data of shape (N, 29)**.
4. **Frame count N:** N is varied from **15 to 64**; the **optimal N** is chosen so that one threshold works for all four attack types and minimizes an error-rate estimate (ERE).
5. **Optimal N:** Found to be **N = 40** (same threshold for DoS, Fuzzy, Gear, RPM, and minimum ERE).

So each training sample is a block of 40 consecutive 29-bit CAN IDs; the model learns to reconstruct these blocks; at inference, reconstruction error is compared to a single threshold.

### 2.4 Training and Model (Base Paper)

- **Model:** **Autoencoder**
  - **Encoder:** Flatten `(N, 29)` → `N×29` units → Dense → **64 units** (ReLU).
  - **Decoder:** Dense 64 → `N×29` (sigmoid) → Reshape to `(N, 29)`.

- **Training:**
  - **Data:** Only **normal** CAN ID sequences (no attack data).
  - **Loss:** Mean Squared Error (MSE) between input and reconstruction.
  - **Optimizer:** Adam, learning rate 0.001.

- **Threshold and N:**
  - **Gaussian Kernel Density Estimation (KDE)** is applied to reconstruction **loss** values:
    - One KDE for “normal” windows (no attack in the window).
    - One KDE per attack type for windows that contain at least one attack frame.
  - **Threshold** for each (normal vs. attack type) is the point where the two KDE curves **intersect**:  
    `Threshold = argmin_x |KDE_normal(x) − KDE_attack(x)|`.
  - The paper searches N in [15, 64] and keeps N for which the **same** threshold works for **all four** attack types. Among those, it selects **N with smallest ERE** (Error Rate Estimation). This yields **N = 40** and a **single** threshold **Th_opt**.

- **Inference:** For a new window of N CAN IDs, compute MSE; if MSE > threshold → attack, else normal.

- **Hardware:** Model is quantized to **16-bit fixed-point**, PLAN sigmoid approximation; implemented on FPGA (e.g., Nexys Video) with ARM Cortex-M3 and CAN controller; validated under real-time CAN traffic.

### 2.5 Why the Base Paper Uses This Model

- **Unsupervised + normal-only training:** Enables detection of **unknown** attack types (not seen in training), unlike supervised methods that only recognize trained classes.
- **Autoencoder:** Normal traffic reconstructs well (low MSE); attacks disturb the learned pattern → higher MSE, so reconstruction error is a natural anomaly score.
- **Sequence of CAN IDs (N frames):** Uses the **order** of IDs on the bus; DoS, fuzzy, and spoofing attacks alter this order or distribution, so sequence-based reconstruction captures more than single-frame or frequency-only methods.
- **Single threshold (KDE):** Avoids per-attack thresholds and multiple models, reducing deployment cost and hardware/software complexity while still covering four attack types.
- **Lightweight (few dense layers, 64-dim bottleneck):** Small parameter count and FLOPs, suitable for FPGA and low power; paper reports large reductions in LUTs, flip-flops, and power vs. other FPGA IDSs.
- **KDE for threshold:** Non-parametric, adapts to the shape of loss distributions for normal vs. attack, making a single threshold feasible across different attack types.

**Reported performance (paper):** Average accuracy 99.2%, precision 99.2%, recall 99.1%, F1 99.2%; after hardware deployment, similar metrics (e.g. average accuracy 99.21%).

---

## Summary Comparison

| Aspect            | Our Solution (Full: LSTM + AE)     | Base Paper                          |
|------------------|------------------------------------|-------------------------------------|
| **Paradigm**      | Hybrid (supervised + unsupervised) | Unsupervised                         |
| **Input**         | One frame: CAN_ID, DLC, DATA0–7 (10) | N consecutive CAN IDs (29-bit each)  |
| **Stage 1**       | LSTM classifier (5 classes)        | —                                   |
| **Stage 2**       | Autoencoder on “Normal” path       | Single autoencoder                  |
| **Output**        | Known class or “Unknown Attack”    | Normal vs. attack (single threshold) |
| **Threshold**     | mean + 3×std (normal train loss)   | KDE intersection (optimal N=40)     |
| **Dataset**       | Car-Hacking (normal TXT + CSVs)    | Same dataset family                  |
| **Preprocessing** | MinMaxScaler; (1,10) for LSTM; 10 for AE | Zero-pad ID to 29 bits; group N=40   |
| **Strengths**     | Known-attack labels + unknown flag | Single model; FPGA-optimized; KDE    |
| **Unknown attacks**| AE on LSTM “Normal” path           | Unsupervised AE on full stream      |

This file documents the **full** solution in `model2025_cursor.py` (Stage 1 + Stage 2) and the base paper for reports or further refinement.
