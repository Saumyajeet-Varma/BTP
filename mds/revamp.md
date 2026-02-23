# Revamp Pipeline: Normal-First CAN Intrusion Detection

This document describes the **revamped** pipeline implemented in `codes/model2025_revamp.py`. It swaps the order of stages compared to `model2025_cursor.py`: **Stage 1** decides whether a message is **Normal** or **Not Normal**; **Stage 2** (only when Not Normal) classifies the attack type (DoS, Fuzzy, Gear, RPM).

---

## 1. Rationale: Why Normal First?

- **Detecting normal messages is more important than detecting the attack.**  
  In many CAN security settings, the majority of traffic is normal. Correctly identifying “normal” reduces false alarms and avoids unnecessary escalation. Misclassifying normal as attack (false positives) is costly; missing an attack (false negative) is also critical. Prioritizing a clear “normal vs not normal” decision first aligns with this.

- **Stage order in the original pipeline (`model2025_cursor.py`):**  
  - Stage 1: LSTM multi-class (Normal, DoS, Fuzzy, Gear, RPM).  
  - Stage 2: Autoencoder on normal only, used only to **re-verify** when LSTM says “Normal” (and to flag “Unknown Attack” if reconstruction is high).

- **Stage order in the revamp:**  
  - **Stage 1:** **Normal vs Not Normal** (autoencoder trained on normal only + threshold).  
  - **Stage 2:** **Attack-type classifier** (LSTM on DoS, Fuzzy, Gear, RPM only), run only when Stage 1 says “Not Normal.”

So we **first** answer “Is this normal?” and **then**, only for non-normal traffic, “What kind of attack?”

---

## 2. Pipeline Overview

```
Raw data (normal TXT + attack CSVs)
    → Same parsing & feature set as model2025_cursor (CAN_ID, DLC, DATA0–7)
    → Train/test split (stratified, 80/20)
    → MinMaxScaler fit on train; reshape for LSTM (samples, 1, N_FEATURES)

Stage 1 — Normal vs Not Normal
    → Autoencoder trained ONLY on normal train samples (reconstruction MSE).
    → Threshold = mean(reconstruction MSE on normal train) + 3×std.
    → At inference: if reconstruction MSE ≤ threshold → predict "Normal" (done).
    → If reconstruction MSE > threshold → "Not Normal" → pass to Stage 2.

Stage 2 — Attack-type classification
    → LSTM classifier trained ONLY on attack train samples (DoS, Fuzzy, Gear, RPM).
    → Input: samples that Stage 1 flagged as Not Normal.
    → Output: one of DoS, Fuzzy, Gear, RPM.

Final prediction
    → Normal  → from Stage 1.
    → DoS / Fuzzy / Gear / RPM → from Stage 2 (only when Stage 1 says Not Normal).
```

---

## 3. Data and Preprocessing

- **Dataset:** Same as `model2025_cursor.py` — [Car-Hacking Dataset](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset).
- **Sources:** Normal from `normal_run_data/normal_run_data.txt`; attacks from `dos_attack.csv`, `fuzzy_attack.csv`, `gear_spoofing.csv`, `rpm_spoofing.csv`.
- **Features:** `CAN_ID`, `DLC`, `DATA0`–`DATA7` (same 11 features).
- **Labels:** Normal, DoS, Fuzzy, Gear, RPM.
- **Preprocessing:** MinMaxScaler fit on train; same reshape `(N, 1, N_FEATURES)` for LSTM where needed.

---

## 4. Stage 1: Normal vs Not Normal (Autoencoder)

| Component   | Description |
|------------|-------------|
| **Input**  | One sample: 11 features (scaled), flattened (no sequence for AE). |
| **Model**  | Autoencoder: Dense(64) → BN → Dense(32) → Dense(16) → Dense(32) → Dense(64) → Dense(11). |
| **Training** | Only **normal** train samples; input = target (reconstruction). MSE loss, Adam, EarlyStopping on `val_loss`. |
| **Threshold** | `mean(MSE on normal train) + 3×std(MSE on normal train)`. |
| **Decision** | MSE ≤ threshold → **Normal**. MSE > threshold → **Not Normal** → send to Stage 2. |

The autoencoder learns the distribution of **normal** traffic; deviations (e.g. attacks) yield higher reconstruction error.

---

## 5. Stage 2: Attack-Type Classifier (LSTM)

| Component   | Description |
|------------|-------------|
| **Input**  | Samples that Stage 1 classified as Not Normal. Shape `(batch, 1, N_FEATURES)`. |
| **Training data** | Only **attack** train samples (DoS, Fuzzy, Gear, RPM) — no Normal. |
| **Model**  | LSTM(128) → BN → Dropout(0.4) → Dense(64) → Dropout(0.3) → Dense(32) → Dense(4, softmax). |
| **Output** | One of: DoS, Fuzzy, Gear, RPM. |
| **Training** | Categorical cross-entropy, class weights for imbalance, EarlyStopping. |

Stage 2 is never run on samples that Stage 1 labels as Normal, which keeps the “normal first” design and reduces unnecessary computation on the majority class.

---

## 6. Inference: `revamp_predict(sample, scaled=True)`

1. Scale the sample if `scaled=False` (using the same scaler as in training).
2. **Stage 1:** Compute reconstruction MSE with the autoencoder.  
   - If MSE ≤ threshold → return **"Normal"**.  
   - If MSE > threshold → continue.
3. **Stage 2:** Reshape sample to `(1, 1, N_FEATURES)`, run through the attack LSTM, take argmax → return **"DoS"**, **"Fuzzy"**, **"Gear"**, or **"RPM"**.

No “Unknown Attack” branch is implemented in the revamp; Stage 2 always outputs one of the four known attack types. You can extend the pipeline later (e.g. low confidence → “Unknown Attack”) if needed.

---

## 7. Comparison with `model2025_cursor.py`

| Aspect | model2025_cursor.py | model2025_revamp.py |
|--------|--------------------|---------------------|
| **Stage 1** | LSTM multi-class (Normal + 4 attacks) | Autoencoder: Normal vs Not Normal |
| **Stage 2** | Autoencoder used only when LSTM says Normal (to confirm or flag Unknown Attack) | LSTM attack-type classifier (DoS, Fuzzy, Gear, RPM) only when Stage 1 says Not Normal |
| **Priority** | Classify everything in one shot; then refine “Normal” with AE | Decide “Normal” first; only then classify attack type |
| **Normal decision** | LSTM + optional AE check | Pure autoencoder + threshold |
| **Attack-type decision** | From LSTM directly | From LSTM only on Stage-1 “Not Normal” samples |

---

## 8. Summary

- **Revamp pipeline:** Stage 1 = Normal vs Not Normal (autoencoder); Stage 2 = Attack type (LSTM on DoS, Fuzzy, Gear, RPM).
- **Rationale:** Treat “is it normal?” as the primary question; attack typing is secondary and only for non-normal traffic.
- **Files:** Implementation in `codes/model2025_revamp.py`; pipeline and rationale documented in `mds/revamp.md`.
