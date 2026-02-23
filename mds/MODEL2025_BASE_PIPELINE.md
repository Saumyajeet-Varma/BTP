# Model and Pipeline: `model2025_base.py`

This document describes the **model** and **pipeline** implemented in `codes/model2025_base.py`, which follows the base paper *"Adaptive Autoencoder-Based Intrusion Detection System with Single Threshold for CAN Networks"* (Kim et al., Sensors 2025, 25, 4174) using the **Car-Hacking Dataset** from [HCRL](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset).

---

## 1. Overview

- **Goal:** Binary intrusion detection for CAN traffic: classify a **window** of consecutive CAN frames as **normal** or **attack**.
- **Approach:** Unsupervised. The model is trained **only on normal** CAN traffic. Attack is detected when **reconstruction error** (MSE) of a window exceeds a **single threshold**.
- **Input:** **CAN ID only** (no DLC, no data payload). Each ID is encoded as **29 bits**; **N** consecutive IDs form one sample of shape **(N, 29)**.
- **Output:** For each window, one decision: **normal** (MSE ≤ threshold) or **attack** (MSE > threshold).

---

## 2. Pipeline (High-Level)

```
Raw logs (normal TXT + attack CSVs)
    → Extract CAN ID per frame
    → Encode each ID to 29 bits (zero-padding)
    → Build sliding windows of N frames → (num_windows, N, 29)
    → Split: normal 80% train / 20% test; attack 2/3 threshold / 1/3 test
    → Train autoencoder on NORMAL TRAIN only (reconstruction)
    → Compute reconstruction MSE on normal train + each attack type (2/3)
    → KDE on normal vs each attack → find intersection threshold per type
    → If same threshold for all 4 types: use it; else fallback (mean+3×std)
    → Evaluate on test set: predict attack if MSE > threshold
```

---

## 3. Data Loading (Car-Hacking Dataset)

**Source:** [Car-Hacking Dataset](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset) (HCRL).

**Attributes per row:** `Timestamp`, `CAN ID` (HEX), `DLC`, `DATA[0]`–`DATA[7]`, `Flag`.

- **Flag:** `T` = injected (attack) message, `R` = normal message (per dataset 1.1).
- Only **CAN ID** is used in the model; DLC and DATA are loaded but not used.

**Files used:**

| Type   | File(s) |
|--------|---------|
| Normal | `normal_run_data/normal_run_data.txt` or `normal_run_data.txt` (attack-free log) |
| DoS    | `dos_attack.csv` |
| Fuzzy  | `fuzzy_attack.csv` |
| Gear   | `gear_spoofing.csv` |
| RPM    | `rpm_spoofing.csv` |

**Loading logic:**

- **Normal:** Parse TXT with regex; extract CAN ID (HEX → int); preserve order → one long sequence of IDs.
- **Attack CSVs:** Read CSV; convert CAN ID from HEX to int; interpret `Flag`: `T` → attack frame, `R` → normal frame. Order is preserved so that sliding windows can be labeled correctly.

---

## 4. Preprocessing

### 4.1 CAN ID → 29-bit encoding

- Standard CAN uses 11-bit IDs; the paper uses **29-bit** representation (compatible with extended CAN 2.0B).
- **Zero-padding:** 11-bit ID is left-padded with 18 zeros → 29 bits, MSB first.
- Each ID becomes a vector of length **29** with values 0 or 1 (stored as `float32`).

### 4.2 Windowing

- **Input:** A sequence of CAN IDs (from one file or one concatenated stream).
- **Parameters:** `N` (e.g. 40) = frames per window, `step` (e.g. 1) = stride.
- **Output:** Array of shape **(num_windows, N, 29)**.
- For **attack** data, each window is labeled: **attack** if **at least one** of the N frames has `Flag == T`; otherwise the window is still “normal” within that attack file (and we only use windows that contain at least one attack frame for threshold/test).

### 4.3 Splits

- **Normal:** All normal windows → **80% train**, **20% test** (random, stratified by window index).
- **Attack (each type):** Only windows that contain ≥1 attack frame → **2/3** for **threshold selection** (and KDE/ERE), **1/3** for **test**.
- The autoencoder is trained **only on normal train** windows. No attack data is used in training.

---

## 5. Model: Autoencoder

**Architecture (paper Section 3.3):**

| Layer        | Shape / operation |
|-------------|--------------------|
| Input       | `(batch, N, 29)`   |
| Flatten     | `(batch, N×29)`    |
| Dense       | 64 units, ReLU    |
| Dense       | `N×29` units, sigmoid |
| Reshape     | `(batch, N, 29)`   |
| Output      | Reconstruct input  |

- **Loss:** Mean Squared Error (MSE) between input and output.
- **Optimizer:** Adam (default).
- **Training:** Input = target (reconstruction). Only **normal train** windows are used. Validation split 10%; EarlyStopping on `val_loss` (patience 5, restore best weights).

So the model learns to compress and reconstruct **sequences of 29-bit CAN IDs** of length N. Normal traffic reconstructs well (low MSE); windows containing attack frames tend to have higher MSE.

---

## 6. Threshold: KDE and ERE

### 6.1 Single threshold via KDE

- For **normal (train)** windows: compute reconstruction MSE per window → distribution of “normal” losses.
- For **attack (threshold portion)** windows of each type: compute MSE per window → distribution of “attack” losses.
- **Gaussian Kernel Density Estimation (KDE)** is applied to both distributions.
- **Threshold** for that attack type = value of **x** where the two density curves **intersect**:  
  `Threshold = argmin_x |KDE_normal(x) − KDE_attack(x)|`.

The paper seeks a **single** threshold that works for **all four** attack types. So:

- Thresholds are computed for DoS, Fuzzy, Gear, RPM.
- If these four thresholds are (approximately) the **same** (e.g. within 5% of their range), that common value is used as the **single threshold**.
- If not, the script falls back to a **statistical** threshold: **mean + 3×std** of the normal train reconstruction MSE.

### 6.2 ERE (Error Rate Estimation)

- **ERE** (paper Equation 8) estimates misclassification rate at a given threshold **LTh**:
  - P(normal misclassified as attack) = integral of KDE_normal from LTh to ∞  
  - P(attack misclassified as normal) = integral of KDE_attack from −∞ to LTh  
  - **ERE = sum of these two** (approximated via numerical integration).

- ERE is used only when **DO_FULL_N_SEARCH = True**: among all **N** in [15, 64] for which the same threshold exists for all four types, the **N** with the **smallest average ERE** (over the four attack types) is chosen. Default is **N = 40** (paper’s reported optimal).

---

## 7. Training Flow

1. **Build splits** for chosen **N** (and step): normal train/test, attack threshold/test per type.
2. **Build autoencoder** for that N: `(N, 29)` → Flatten → Dense(64, ReLU) → Dense(N×29, sigmoid) → Reshape.
3. **Train** autoencoder on **normal train** windows only (MSE, Adam, EarlyStopping).
4. **Compute MSE** on normal train and on each attack type’s threshold portion.
5. **KDE** on those losses; **find threshold** (intersection) per attack type.
6. If **same threshold** for all four → use it; else use **mean + 3×std** on normal train MSE.
7. (Optional) If **DO_FULL_N_SEARCH**: repeat for N = 15,…,64 and pick N (and its threshold) with smallest average ERE.

---

## 8. Inference and Evaluation

- **Per window:** Forward pass through the autoencoder → reconstruction → MSE.
- **Decision:** If MSE **> threshold** → **attack**; else → **normal**.
- **Test set:** Held-out normal windows (20%) + held-out attack windows (1/3 per type). All are labeled binary: normal (0) or attack (1).
- **Metrics:** Accuracy, precision, recall, F1-score, and confusion matrix (TN, FP, FN, TP).

---

## 9. Summary Table

| Component      | Description |
|----------------|-------------|
| **Data**       | Car-Hacking Dataset; CAN ID only; Flag T/R for attack vs normal. |
| **Input shape**| One sample = **(N, 29)** (N consecutive 29-bit CAN IDs). |
| **Model**      | Autoencoder: Flatten → Dense(64, ReLU) → Dense(N×29, sigmoid) → Reshape. |
| **Training**   | Unsupervised; only normal windows; MSE loss. |
| **Threshold**  | Single value from KDE (intersection of normal vs attack loss distributions); fallback: mean + 3×std. |
| **Output**     | Binary: normal or attack per window. |
| **N**          | Default 40; optional search 15–64 via ERE. |

This pipeline matches the base paper’s design and uses the same data source as your other solutions (`model2025_cursor.py`), with input and threshold logic as in Kim et al., Sensors 2025.
