# Model 2025 Cursor — Explanation for Panel Presentation

This document explains **`model2025_cursor.py`**: the pipeline, dataset, model design, and rationale, in a form suitable for presenting to a panel.

---

## 1. Executive Summary

**What it is:** A **two-stage hybrid Intrusion Detection System (IDS)** for Controller Area Network (CAN) bus traffic in vehicles.

**What it does:**
- **Stage 1:** Classifies each CAN message into one of five **known** classes: **Normal**, **DoS**, **Fuzzy**, **Gear** (gear spoofing), or **RPM** (RPM spoofing).
- **Stage 2:** When Stage 1 predicts **Normal**, an autoencoder checks whether the message is truly normal or an **unknown attack**. If reconstruction error is high, the message is labeled **Unknown Attack**.

**Why it matters:** It combines **supervised** classification of known attacks with **unsupervised** detection of anomalies, so the system can both label known attack types and flag novel attacks that were not seen during training.

---

## 2. Dataset

### 2.1 Source

- **Name:** Car-Hacking Dataset  
- **Provider:** Hacking and Countermeasure Research Lab (HCRL)  
- **Link:** https://ocslab.hksecurity.net/Datasets/car-hacking-dataset  

### 2.2 Contents

| Data type | Source file | Description |
|-----------|-------------|-------------|
| **Normal** | `normal_run_data/normal_run_data.txt` | Attack-free CAN traffic (parsed from TXT) |
| **DoS** | `dos_attack.csv` | Denial-of-Service attack |
| **Fuzzy** | `fuzzy_attack.csv` | Fuzzy (random) attack |
| **Gear** | `gear_spoofing.csv` | Gear spoofing attack |
| **RPM** | `rpm_spoofing.csv` | RPM spoofing attack |

### 2.3 Schema (per CAN message)

- **Timestamp** — Message timing  
- **CAN_ID** — Identifier (11-bit, stored as integer)  
- **DLC** — Data Length Code (0–8 bytes)  
- **DATA0–DATA7** — Up to 8 data bytes (zero-padded if DLC &lt; 8)  
- **Label** — Normal, DoS, Fuzzy, Gear, or RPM (we add this when loading)

### 2.4 Why This Dataset?

- Publicly available, widely used in automotive security research.  
- Contains both **normal** and **multiple attack types** (DoS, fuzzy, spoofing), so we can train supervised classification and compare with baseline methods.  
- Same dataset family as the **base paper** (Kim et al., *Sensors* 2025) we compare against, ensuring a fair comparison.

---

## 3. Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RAW DATA (Car-Hacking Dataset)                       │
│  normal_run_data.txt  +  dos_attack.csv  +  fuzzy_attack.csv  +  gear/rpm   │
└─────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PREPROCESSING                                                               │
│  • Parse TXT (normal) and CSV (attacks)                                      │
│  • Features: CAN_ID, DLC, DATA0–DATA7 (10 features per message)             │
│  • Stratified 80/20 train/test split                                        │
│  • MinMaxScaler (fit on train only — no leakage)                            │
│  • Reshape for LSTM: (samples, 1, 10)                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1 — LSTM CLASSIFIER (Supervised)                                      │
│  • Input: one CAN frame (1 time step × 10 features)                          │
│  • Output: one of 5 classes — Normal, DoS, Fuzzy, Gear, RPM                 │
│  • Class weights used to handle imbalanced classes                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    │  Predicted ≠ Normal?                       │
                    │  (DoS / Fuzzy / Gear / RPM)                │
                    └─────────────────────┬─────────────────────┘
                      YES ▼                              ▼ NO (Normal)
              ┌───────────────────┐           ┌───────────────────────────────┐
              │  Return that      │           │  STAGE 2 — AUTOENCODER        │
              │  class            │           │  • Trained only on Normal      │
              └───────────────────┘           │  • Reconstruction MSE vs      │
                                              │    threshold (mean + 3×std)   │
                                              └───────────────┬───────────────┘
                                                               │
                                    MSE > threshold? ─── YES ──► "Unknown Attack"
                                    MSE ≤ threshold? ─── NO  ──► "Normal"
```

---

## 4. Pipeline Stages in Detail

### 4.1 Exploratory Data Analysis (EDA)

Before modeling, the script performs EDA and plots:

- Class distribution (message counts per label)  
- Top CAN IDs by frequency  
- DLC distribution (0–8 bytes)  
- Correlation heatmap of the 10 features  
- DATA byte distributions by class (box plots)  
- Top CAN IDs per class  

These help the panel understand the data and justify the need for class weights and the chosen features.

### 4.2 Preprocessing

1. **Features:** `['CAN_ID', 'DLC', 'DATA0', 'DATA1', 'DATA2', 'DATA3', 'DATA4', 'DATA5', 'DATA6', 'DATA7']` — **10 features** per message.  
2. **Labels:** `LabelEncoder` maps class names to integers; `to_categorical` produces one-hot vectors for the LSTM.  
3. **Split:** Stratified 80% train / 20% test (same random seed for reproducibility).  
4. **Scaling:** `MinMaxScaler` fitted **only on the training set**, then applied to train and test.  
5. **LSTM input shape:** `(number_of_samples, 1, 10)` — one time step per message, 10 features.  
6. **Class imbalance:** `compute_class_weight('balanced', ...)` on training labels; weights are passed to the LSTM’s `fit()`.

### 4.3 Stage 1 — LSTM Classifier

- **Role:** Supervised multi-class classification of **known** attack types and normal traffic.  
- **Architecture:**  
  - Input `(batch, 1, 10)` → **LSTM(128)** → **BatchNormalization** → **Dropout(0.4)** → **Dense(64, ReLU)** → **Dropout(0.3)** → **Dense(32, ReLU)** → **Dense(5, softmax)**.  
- **Training:** Adam, categorical cross-entropy, up to 30 epochs, batch size 64, 10% validation split, **class weights**, **EarlyStopping** on validation loss (patience 5, restore best weights).  
- **Output:** One of **Normal**, **DoS**, **Fuzzy**, **Gear**, **RPM**.

### 4.4 Stage 2 — Autoencoder (Anomaly Detection)

- **Role:** When the LSTM predicts **Normal**, the autoencoder decides if the message is truly normal or an anomaly (**Unknown Attack**).  
- **Training data:** **Only normal** training samples (no attack data). The autoencoder learns to reconstruct “normal” CAN frames.  
- **Architecture:**  
  - Input `(batch, 10)` → Dense(64, ReLU) → BatchNorm → Dense(32, ReLU) → Dense(16, ReLU) → Dense(32, ReLU) → Dense(64, ReLU) → Dense(10, linear).  
- **Training:** Input = target (reconstruction). Adam, **MSE** loss, same epoch/batch/validation/EarlyStopping strategy as Stage 1.  
- **Threshold:** On **normal training** samples, compute reconstruction MSE; then  
  **threshold = mean(MSE) + 3 × std(MSE)**.  
- **Inference:** If LSTM says Normal and reconstruction MSE **> threshold** → **Unknown Attack**; else **Normal**.

### 4.5 Hybrid Prediction (Inference)

For each test sample:

1. Scale with the same fitted `MinMaxScaler` (if not already scaled).  
2. Run **Stage 1 (LSTM)**.  
3. If predicted class is **not** Normal → return that class (DoS, Fuzzy, Gear, RPM).  
4. If predicted class is **Normal** → run **Stage 2 (autoencoder)**.  
   - If MSE > threshold → return **Unknown Attack**.  
   - Else → return **Normal**.

So the **final output** is one of: **Normal**, **DoS**, **Fuzzy**, **Gear**, **RPM**, **Unknown Attack**.

---

## 5. Why We Use This Model

| Goal | How this model addresses it |
|------|-----------------------------|
| **Classify known attacks** | Stage 1 LSTM uses full frame (ID, DLC, data) for strong multi-class classification of Normal, DoS, Fuzzy, Gear, RPM. |
| **Handle class imbalance** | Class weights in the LSTM loss balance underrepresented attack types. |
| **Detect unknown attacks** | Stage 2 autoencoder is trained only on normal traffic; messages that look “Normal” to the LSTM but are anomalous get high reconstruction error and are labeled **Unknown Attack**. |
| **Single, simple threshold** | One scalar (mean + 3×std on normal training MSE) for Stage 2 — no per-attack thresholds. |
| **Unified feature set** | Same 10 features and same scaler for both stages — simpler pipeline and easier deployment. |
| **Alignment with literature** | Same dataset family as the base paper (Kim et al., *Sensors* 2025); our model extends the idea with a **hybrid** design: supervised labels for known attacks + unsupervised anomaly detection for unknowns. |

---

## 6. Evaluation and Visualizations

The script reports and plots:

1. **Class distribution** in the dataset.  
2. **LSTM training curves** — loss and accuracy (train vs validation).  
3. **Autoencoder training curve** — reconstruction loss (train vs validation).  
4. **Hybrid confusion matrix** — true vs predicted labels (including **Unknown Attack**).  
5. **Reconstruction error histogram** — Normal vs Attack (test set) and the threshold line.  
6. **Classification report** — precision, recall, F1 for all classes including Unknown Attack.  
7. **Hybrid accuracy** — overall accuracy of the two-stage pipeline on the test set.

These support the panel discussion on performance and the value of the second stage.

---

## 7. Key Points for the Panel

1. **Dataset:** Car-Hacking Dataset (HCRL); same family as the *Sensors* 2025 baseline; normal + four attack types (DoS, Fuzzy, Gear, RPM).  
2. **Pipeline:** Two-stage hybrid — **LSTM (supervised)** for known classes + **autoencoder (unsupervised)** for anomalies when LSTM predicts Normal.  
3. **Features:** 10 per message — CAN_ID, DLC, DATA0–DATA7; MinMax-scaled; no scaling leakage (scaler fit on train only).  
4. **Why hybrid:** Combines clear labels for known attacks with the ability to flag **unknown** attacks that the classifier might wrongly call Normal.  
5. **Reproducibility:** Fixed random seed (42) for NumPy, TensorFlow, and train/test split.  
6. **Deployment note:** Same features and scaling for both stages simplify integration; threshold is a single number derived from normal training data.

---

## 8. Reference

- **Base paper (baseline comparison):** Kim, D.; Im, H.; Lee, S. “Adaptive Autoencoder-Based Intrusion Detection System with Single Threshold for CAN Networks.” *Sensors* 2025, 25, 4174. https://doi.org/10.3390/s25134174  
- **Dataset:** Car-Hacking Dataset — https://ocslab.hksecurity.net/Datasets/car-hacking-dataset  

---

*This document summarizes `model2025_cursor.py` for panel presentation. For implementation details, see the script and the project’s `SOLUTION_FULL_AND_BASELINE.md` and `DIFFERENCE_STAGE1_VS_FULL.md`.*
