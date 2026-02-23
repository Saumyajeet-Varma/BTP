# CAN Intrusion Detection: Our Solution vs. Base Paper

This document describes **our solution** (Stage 1 LSTM classifier) and the **base paper** (adaptive autoencoder-based IDS) in terms of solution overview, dataset, preprocessing, training/model, and rationale for the chosen approach.

---

## Part 1 — Our Solution (Stage 1 LSTM Classifier)

### 1.1 Solution Overview

Our solution is a **supervised**, **multi-class** intrusion detection system for Controller Area Network (CAN) traffic. It uses a single **LSTM-based classifier** that takes one CAN message at a time and predicts the class: **Normal**, **DoS**, **Fuzzy**, **Gear** (gear spoofing), or **RPM** (RPM spoofing). The pipeline is implemented as **Stage 1 only**; a planned Stage 2 (e.g., autoencoder) is not included in the current code.

- **Learning paradigm:** Supervised (labels: Normal + four attack types).
- **Input unit:** One CAN frame (one row) with CAN_ID, DLC, and 8 data bytes.
- **Output:** Probability distribution over 5 classes; final decision is the class with maximum probability.
- **Use case:** Classify each incoming CAN message in real time as normal or as one of four known attack types.

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
  - `Timestamp`, `CAN_ID`, `DLC`, `DATA0`–`DATA7` (8 data bytes), and a `Flag` column in CSVs; we add a `Label` (Normal, DoS, Fuzzy, Gear, RPM).

- **Usage:** All five classes are concatenated into one DataFrame. The same dataset family as the base paper (car-hacking dataset) is used, but we use **full frame fields** (CAN_ID, DLC, DATA0–7) rather than CAN ID only.

### 1.3 Dataset Preprocessing

1. **Parsing**
   - **Normal:** Regex parsing of the TXT file to extract Timestamp, CAN_ID (hex → int), DLC, and DATA; DATA is zero-padded to 8 bytes.
   - **Attacks:** CSV read with columns `Timestamp`, `CAN_ID`, `DLC`, `DATA0`–`DATA7`, `Flag`; a `Label` column is set per file (DoS, Fuzzy, Gear, RPM).

2. **Feature set**
   - **Features used for modeling:** `['CAN_ID', 'DLC', 'DATA0', 'DATA1', 'DATA2', 'DATA3', 'DATA4', 'DATA5', 'DATA6', 'DATA7']`  
   - So we use **10 features** per message (1 ID + 1 DLC + 8 data bytes).

3. **Train/validation/test**
   - **Stratified** 80/20 train–test split on the full labeled dataset (all five classes).
   - **Scaler:** `MinMaxScaler` fitted **only on the training set**; applied to both train and test (no leakage).

4. **Reshaping for LSTM**
   - Each sample is shaped as `(1, N_FEATURES)` i.e. `(1, 10)`:
     - One time step, 10 features per step.
   - So the model sees each CAN message as a short sequence of length 1 with 10 features, which allows using the same LSTM interface for potential future sequence-length extensions.

5. **Labels**
   - `LabelEncoder` for class names → integers; then `to_categorical` for one-hot targets (categorical cross-entropy).

6. **Class imbalance**
   - `compute_class_weight('balanced', ...)` on the training labels and passed to `model.fit(class_weight=class_weights)` to weight loss by inverse class frequency.

### 1.4 Training and Model

- **Architecture (Stage 1 — LSTM classifier):**
  - **Input:** `(batch, 1, 10)` — one time step, 10 features.
  - **LSTM:** 128 units, `return_sequences=False`.
  - **BatchNormalization** → **Dropout(0.4)** → **Dense(64, relu)** → **Dropout(0.3)** → **Dense(32, relu)** → **Dense(5, softmax)**.

- **Training setup:**
  - **Optimizer:** Adam (default).
  - **Loss:** `categorical_crossentropy`.
  - **Metrics:** Accuracy.
  - **Epochs:** Up to 30, with **EarlyStopping** on `val_loss` with `patience=5` and `restore_best_weights=True`.
  - **Batch size:** 64.
  - **Validation:** 10% of training data as validation split.
  - **Class weights:** Applied to compensate for imbalanced classes.

- **Reproducibility:** `RANDOM_STATE=42` for NumPy and TensorFlow seeds; same seed used for `train_test_split`.

### 1.5 Why We Use This Model

- **LSTM:** Captures sequential/temporal structure. Even with a single time step in Stage 1, the same interface can be extended to longer sequences (e.g., windows of consecutive CAN messages) without changing the model type.
- **Supervised multi-class:** We have labeled Normal and four attack types; supervised learning directly fits this setting and typically gives strong accuracy on **known** attack types.
- **Full frame (ID + DLC + data):** Uses more information than ID-only approaches, which can help distinguish attacks that alter payload (e.g., gear/RPM spoofing) or DLC.
- **Class weights:** Handles imbalance between normal and attack messages and among attack types.
- **Lightweight Stage 1:** One LSTM + a few Dense layers is relatively small and suitable as a first stage before adding an optional Stage 2 (e.g., autoencoder for anomaly detection).

**Limitation:** This design is aimed at **known** attack classes; it does not explicitly target **unknown** attack types. Unknown attacks would be classified into one of the five trained classes (or misclassified), unlike the base paper’s unsupervised approach.

---

## Part 2 — Base Paper (Adaptive Autoencoder-Based IDS with Single Threshold)

**Reference:** Kim, D.; Im, H.; Lee, S. “Adaptive Autoencoder-Based Intrusion Detection System with Single Threshold for CAN Networks.” *Sensors* 2025, 25, 4174.  
**DOI:** https://doi.org/10.3390/s25134174  

### 2.1 Solution Overview (Base Paper)

The base paper proposes a **lightweight, unsupervised** IDS for CAN networks, designed for **real-time, on-device** deployment (including FPGA):

- **Learning paradigm:** Unsupervised. The model is trained **only on normal** CAN traffic.
- **Core idea:** An **autoencoder** learns to reconstruct “normal” CAN ID sequences. Attack traffic yields **higher reconstruction error**; a **single threshold** (derived via Gaussian KDE) separates normal from attack for **all four attack types**.
- **Main contributions:**
  1. Unsupervised autoencoder so that **unseen** attack types can still be flagged as anomalies.
  2. **Single threshold** for all attack types (unlike e.g. NovelADS, which uses per-attack thresholds).
  3. **Lightweight** design: few parameters and low FLOPs, validated on FPGA with reduced LUTs, flip-flops, and power vs. existing FPGA-based IDSs.

### 2.2 Dataset (Base Paper)

- **Source:** Vehicle hacking dataset from the **Hacking and Countermeasure Research Lab** (same family as our dataset).  
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
  - **Training:** Only **normal** frames.
  - **Threshold/frame-count selection:** Two-thirds of each attack type’s data (with labels) used to compute KDE and optimal threshold and N.
  - **Test:** Remaining one-third of attack data (and normal) for final evaluation; attack data are **not** used in training.

### 2.3 Dataset Preprocessing (Base Paper)

1. **Input used:** **CAN ID only** (no DLC or data payload in the model input).
2. **Bit representation:** CAN IDs are expanded to **29 bits** using **zero-padding** (compatible with CAN 2.0A/2.0B).
3. **Sequence construction:** Consecutive CAN IDs are grouped into **N frames** per sample, forming **2D data of shape (N, 29)**.
4. **Frame count N:** N is varied from **15 to 64**; 50 configurations are used to find the **optimal N** (chosen so that one threshold works for all four attack types and minimizes error rate).
5. **Optimal N:** Found to be **N = 40** (same threshold for DoS, Fuzzy, Gear, RPM, and minimum ERE).

So each training sample is a block of 40 consecutive 29-bit CAN IDs, and the model learns to reconstruct these blocks; at inference, reconstruction error is compared to a single threshold.

### 2.4 Training and Model (Base Paper)

- **Model:** **Autoencoder**
  - **Encoder:** Flatten `(N, 29)` → `N×29` units → Dense layer → **64 units** (ReLU).
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
  - The paper searches N in [15, 64] and keeps N for which the **same** threshold works for **all four** attack types. Among those, it selects **N with smallest ERE** (Error Rate Estimation — sum of probabilities of misclassifying normal as attack and attack as normal). This yields **N = 40** and a **single** threshold **Th_opt**.

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

| Aspect            | Our Solution (Stage 1)        | Base Paper                    |
|------------------|-------------------------------|-------------------------------|
| **Paradigm**      | Supervised                    | Unsupervised                  |
| **Input**         | One frame: CAN_ID, DLC, DATA0–7 | N consecutive CAN IDs (29-bit each) |
| **Model**         | LSTM classifier (5 classes)   | Autoencoder (reconstruction)  |
| **Output**        | Class: Normal / DoS / Fuzzy / Gear / RPM | Normal vs. attack (single threshold) |
| **Dataset**       | Car-Hacking (normal TXT + attack CSVs) | Same dataset family (normal + 4 attacks) |
| **Preprocessing** | MinMaxScaler, (1, 10) per sample | Zero-pad ID to 29 bits, group N=40 frames → (40, 29) |
| **Strengths**     | Strong on known attacks; uses full frame | Detects unknown attacks; single threshold; FPGA-friendly |
| **Weakness**      | Does not target unknown attack types   | Slightly lower than supervised on some metrics; threshold from part of attack data |

This markdown file can be used as the main reference for “our solution vs. base paper” in reports or documentation.
