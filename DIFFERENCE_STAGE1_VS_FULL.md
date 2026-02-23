# Difference: `model2025-stage1_cursor.py` vs. `model2025_cursor.py`

This document explains how **Stage 1 only** (`model2025-stage1_cursor.py`) differs from the **full two-stage pipeline** (`model2025_cursor.py`).

---

## At a Glance

| Aspect | `model2025-stage1_cursor.py` | `model2025_cursor.py` |
|--------|------------------------------|------------------------|
| **Pipeline** | Stage 1 only | Stage 1 + Stage 2 |
| **Models** | LSTM classifier | LSTM classifier + Autoencoder |
| **Output classes** | 5: Normal, DoS, Fuzzy, Gear, RPM | 5 + **Unknown Attack** (6 possible predictions) |
| **Unknown-attack handling** | None (all go to one of 5 classes) | Yes (autoencoder flags anomalies as "Unknown Attack") |
| **Evaluation** | LSTM accuracy & confusion matrix (5×5) | Hybrid accuracy & confusion matrix (includes "Unknown Attack") |
| **Visualizations** | Class dist, LSTM curves, confusion matrix | Same + autoencoder curves, reconstruction-error histogram, hybrid confusion matrix |
| **Lines of code** | ~332 | ~443 |

---

## What Is the Same

Both scripts share:

- **Dataset:** Car-Hacking Dataset (normal TXT + four attack CSVs).
- **Features:** `CAN_ID`, `DLC`, `DATA0`–`DATA7` (10 features).
- **Preprocessing:** Same parsing, `MinMaxScaler` (fit on train), stratified 80/20 split, reshape to `(1, 10)` for the LSTM.
- **Stage 1 LSTM:** Same architecture and training:
  - Input `(1, 10)` → LSTM(128) → BatchNorm → Dropout(0.4) → Dense(64) → Dropout(0.3) → Dense(32) → Dense(5, softmax).
  - Adam, categorical cross-entropy, class weights, EarlyStopping (val_loss, patience 5).

So **Stage 1 is identical** in both files; the difference is what happens after the LSTM.

---

## What Is Different

### 1. Stage 2 (Autoencoder) — Only in Full Pipeline

**`model2025_cursor.py`** adds:

- **Stage 2 — Autoencoder** trained only on **normal** training samples (same 10 features, already MinMax-scaled):
  - Architecture: Input(10) → Dense(64, relu) → BatchNorm → Dense(32, relu) → Dense(16, relu) → Dense(32, relu) → Dense(64, relu) → Dense(10, linear).
  - Loss: MSE (reconstruction). Optimizer: Adam. EarlyStopping on val_loss, patience 5.

- **Threshold:** On normal training data, reconstruction MSE is computed; then  
  `threshold = mean(train_loss) + 3 * std(train_loss)`.

**`model2025-stage1_cursor.py`** has no autoencoder and no threshold.

---

### 2. Prediction Logic

**Stage 1 only (`model2025-stage1_cursor.py`):**

- For each sample: run LSTM → take argmax → output one of: **Normal**, **DoS**, **Fuzzy**, **Gear**, **RPM**.
- Every message is forced into one of the five trained classes.

**Full pipeline (`model2025_cursor.py`):**

- **Hybrid prediction** (implemented as `hybrid_predict()`):
  1. Run LSTM. If predicted class ≠ **Normal** → return that class (DoS, Fuzzy, Gear, RPM).
  2. If LSTM predicts **Normal** → run autoencoder on the same (scaled) sample:
     - If reconstruction MSE **> threshold** → return **"Unknown Attack"**.
     - Else → return **"Normal"**.

So the full pipeline can output **six** possible labels: Normal, DoS, Fuzzy, Gear, RPM, and **Unknown Attack**. The Stage-1-only script never outputs "Unknown Attack".

---

### 3. Evaluation and Reported Metrics

**Stage 1 only:**

- Predictions: `classifier.predict(X_test_lstm)` → argmax → class names.
- **Metric:** "Stage 1 (LSTM) Accuracy" = accuracy of these 5-class predictions.
- **Confusion matrix:** 5×5 (true vs. predicted among Normal, DoS, Fuzzy, Gear, RPM).
- **Classification report:** Precision/recall/F1 for the five classes.

**Full pipeline:**

- Predictions: for each test sample, `hybrid_predict(X_test[i], scaled=True)` (LSTM + optional autoencoder).
- **Metric:** "Hybrid Accuracy" = accuracy of these hybrid predictions (including "Unknown Attack" as a predicted label).
- **Confusion matrix:** Can include **"Unknown Attack"** as an extra predicted class (so labels are the five true classes + "Unknown Attack" in the prediction set).
- **Classification report:** For the hybrid predictions (again with "Unknown Attack" as a possible prediction).

---

### 4. Visualizations

**Stage 1 only:**

1. Class distribution in dataset  
2. LSTM training curves (loss, accuracy)  
3. Confusion matrix for **LSTM-only** predictions (5×5)

**Full pipeline:**

1. Class distribution in dataset  
2. LSTM training curves (loss, accuracy)  
3. **Autoencoder** training curve (MSE loss)  
4. **Hybrid** confusion matrix (includes "Unknown Attack")  
5. **Reconstruction error:** histogram of MSE for normal vs. attack test samples, with threshold line  
6. Classification report for **hybrid** predictions  

---

## When to Use Which

- **Use `model2025-stage1_cursor.py`** when you only need **known-attack classification** (Normal + four attack types), with a single LSTM and simpler evaluation. No notion of "unknown" attacks.
- **Use `model2025_cursor.py`** when you want **known-attack classification plus unknown-attack detection**: the LSTM handles the five known classes, and the autoencoder refines "Normal" into Normal vs. "Unknown Attack" using reconstruction error.

In short: **Stage 1 only** = LSTM classifier; **full** = same LSTM + autoencoder gate for "Normal" → Normal or Unknown Attack.
