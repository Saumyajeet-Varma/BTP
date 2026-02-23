# Technical Report: Intrusion Detection System (IDS) Pipeline for CAN Bus Traffic

**Document type:** Technical report  
**Source:** IDS_new_pipeline.ipynb  
**Version:** 1.0  

---

## Abstract

This report describes a two-stage Intrusion Detection System (IDS) pipeline for Controller Area Network (CAN) bus traffic. **Stage 1** classifies traffic into multiple known attack types (DoS, Fuzzy, Gear, RPM). **Stage 2** uses an autoencoder trained on normal data to model normal behaviour and to detect unknown attacks via reconstruction error. The document covers the dataset, data preprocessing, pipeline design, model architecture, training procedure, testing and evaluation, rationale for the design, and future scope and challenges.

---

## 1. Introduction

The pipeline implements a swapped two-stage approach: an LSTM-based multi-class attack classifier (Stage 1) followed by an autoencoder-based normal/abnormal detector (Stage 2). Stage 1 identifies known attack types; Stage 2 refines Normal vs non-normal and supports detection of unknown attacks. The implementation uses the Car-Hacking Dataset and is configurable via a subset option for development and full runs.

---

## 2. Dataset

**Source:** Car-Hacking Dataset (CAN bus traffic).

**Normal traffic:** `normal_run_data/normal_run_data.txt` — text format with parsed fields (Timestamp, CAN_ID, DLC, DATA bytes).

**Attack traffic (CSV):**
- `DoS_dataset.csv`
- `Fuzzy_dataset.csv`
- `gear_dataset.csv`
- `RPM_dataset.csv`

Each attack file includes columns: `Timestamp`, `CAN_ID`, `DLC`, `DATA0`–`DATA7`, and **`Flag`** (used for per-row labeling).

**Subset configuration (for manageable runs):**
- `USE_SUBSET = True`: cap rows to avoid memory/runtime issues.
- `MAX_NORMAL = 10_000` (normal messages).
- `MAX_PER_ATTACK_FILE = 20_000` (per attack CSV).

**Labels:** Five classes — **Normal**, **DoS**, **Fuzzy**, **Gear**, **RPM**.
- Normal file: all rows labeled `"Normal"`.
- Attack files: **Flag-based labeling** — `Flag == 'R'` → `"Normal"`, `Flag == 'T'` → attack name (DoS / Fuzzy / Gear / RPM), so each row is labeled by message type, not by file alone.

**Features used:** `CAN_ID`, `DLC`, `DATA0`, `DATA1`, `DATA2`, `DATA3`, `DATA4`, `DATA5`, `DATA6`, `DATA7` (10 features; `N_FEATURES = 10`).

---

## 3. Data Preprocessing

**Normal data:**
- Parse each line with a regex extracting Timestamp, CAN_ID, DLC, and DATA bytes.
- Expand `DATA` into columns `DATA0`–`DATA7` (pad to 8 bytes if needed).
- Stop after `MAX_NORMAL` rows when `USE_SUBSET` is True.
- Assign `Label = 'Normal'`.

**Attack data:**
- Read CSVs with fixed column names; optionally limit rows via `nrows=MAX_PER_ATTACK_FILE`.
- `convert_numeric_columns`: CAN_ID and DATA* as hex → int; DLC and rest numeric; fill NaN with 0.
- `label_from_flag(Flag, attack_name)`: R → `'Normal'`, T → attack name (e.g. `'DoS'`), so each row is labeled by message type.

**Combining:**
- `full_df = concat(normal, dos, fuzzy, gear, rpm)`.
- Feature matrix `X = full_df[features].values`, labels `y = full_df['Label'].values`.

**Train/validation split:**
- `train_test_split(X, y_cat_full, test_size=0.2, stratify=y_encoded_full, random_state=RANDOM_STATE)`.

**Scaling:**
- `MinMaxScaler` fit on `X_train`, transform both `X_train` and `X_test`.

**Reshape for LSTM:**
- `X_train_lstm` / `X_test_lstm`: shape `(n_samples, 1, N_FEATURES)` for the attack classifier.

---

## 4. Pipeline

The pipeline is **two-stage** and uses a **swapped order** (attack classifier first, then autoencoder):

**Stage 01 — Multiple-attack classifier (run first)**  
- LSTM model trained only on **attack** samples.  
- **Role:** Classifies traffic into one of the **known attack types** — **DoS**, **Fuzzy**, **Gear**, **RPM** (4 classes).

**Stage 02 — Normal profile & unknown-attack detection (run second)**  
- Autoencoder trained only on **normal** traffic.  
- **Role:** Models normal behaviour; high reconstruction error indicates deviation from normal (i.e. attack), including **unknown attacks** not seen in training.  
- A **threshold** (e.g. `mean(train_loss) + 3 * std(train_loss)` on normal training data) separates “Normal” (low recon error) from “Not Normal” (attack or unknown attack).

**Prediction logic (`revamp_predict`):**
1. Run **Stage 01** (LSTM) on the sample → get tentative **attack_type** (DoS/Fuzzy/Gear/RPM).
2. Run **Stage 02** (autoencoder) on the sample → compute reconstruction error.
3. If reconstruction error **≤ threshold** → return **"Normal"** (override LSTM).
4. Else → return the **attack_type** from Stage 01 (known attack) or treat as attack/anomaly (covers unknown attacks).

**Summary:** **Stage 1** classifies **multiple known attacks**; **Stage 2** uses normal data to identify Normal vs non-normal and supports **detecting unknown attacks** via high reconstruction error.

---

## 5. Model Architecture

**Stage 01 — Multiple-attack classifier (LSTM)**  
- Input: `(batch, 1, N_FEATURES)`.  
- LSTM(128, return_sequences=False).  
- BatchNormalization.  
- Dropout(0.4).  
- Dense(64, relu).  
- Dropout(0.3).  
- Dense(32, relu).  
- Dense(4, softmax) → DoS, Fuzzy, Gear, RPM.

**Stage 02 — Autoencoder (normal profile / unknown-attack detection)**  
- Input: `(batch, input_dim)` with `input_dim = 10`.  
- Encoder: Dense(64, relu) → BatchNorm → Dense(32, relu) → Dense(16, relu).  
- Decoder: Dense(32, relu) → Dense(64, relu) → Dense(input_dim, linear).  
- Loss: MSE; metric: MAE.  
- Trained on normal data only; high reconstruction error flags anomalies (including unknown attacks).

**Threshold:**  
- Computed on **training normal** data: reconstruct with autoencoder, then `threshold = mean(MSE per sample) + 3 * std(MSE per sample)`.

---

## 6. Training Procedure

**Reproducibility:** `RANDOM_STATE = 42`, `np.random.seed`, `tf.random.set_seed`.

**Stage 01 (multiple-attack classifier):**  
- Train on attack samples only (`X_train_attack_lstm`, `y_train_attack_cat`).  
- Loss: `categorical_crossentropy`; metrics: `accuracy`.  
- Class weights: `compute_class_weight('balanced', ...)` to handle class imbalance.  
- Epochs: 30; batch_size: 64; validation_split: 0.1.  
- EarlyStopping: monitor `val_loss`, patience=5, restore_best_weights=True.

**Stage 02 (autoencoder):**  
- Train on normal samples only (`X_train_normal`, `X_train_normal`).  
- Loss: MSE; metrics: MAE.  
- Epochs: 30; batch_size: 64; validation_split: 0.1.  
- EarlyStopping: monitor `val_loss`, patience=5, restore_best_weights=True.

**Threshold:**  
- After Stage 02: reconstruct `X_train_normal`, compute per-sample MSE, then set threshold as above.

---

## 7. Testing and Evaluation

**Prediction:** For each test sample, `revamp_predict(sample, scaled=True)` runs Stage 01 (LSTM) then Stage 02 (autoencoder) and applies the threshold to decide Normal vs attack type.

**Metrics:**  
- `accuracy_score(true_labels, revamp_results)`.  
- `confusion_matrix` and `classification_report` (precision, recall, f1-score per class).

**Typical outcome (subset run):** Overall accuracy ~92.78%; high performance on Normal, Gear, RPM; DoS/Fuzzy may need further tuning or more data.

**Note:** Per-sample `revamp_predict` is slow for large test sets; batched evaluation (autoencoder on full `X_test`, then LSTM only where needed) can be used for speed.

---

## 8. Rationale (Why This Model)

- **Two-stage design:** **Stage 1** classifies **multiple known attacks** (DoS, Fuzzy, Gear, RPM); **Stage 2** uses **normal data** to model normal behaviour and to **detect unknown attacks** (high reconstruction error = not normal). This fits CAN traffic where normal is abundant and attack types are distinct.

- **Swapped order (LSTM → Autoencoder):**  
  - Stage 1 (LSTM) sees all samples and gives a tentative known-attack label.  
  - Stage 2 (autoencoder), trained on normal only, corrects to “Normal” when the sample looks normal (low reconstruction error) and flags non-normal traffic (including unknown attacks) when reconstruction error is high.  
  - This improves precision on Normal, reduces false attack labels, and supports detection of unseen attack types.

- **Flag-based labeling:** Uses the dataset’s `Flag` (R/T) so that mixed attack files are labeled per message, improving supervision quality.

- **Subset option:** Allows fast iteration and debugging without loading the full dataset.

---

## 9. Future Scope and Challenges

**Future work:**  
- **Batched evaluation:** Run autoencoder on full `X_test`, then LSTM only for samples above threshold (or vice versa depending on design) to speed up evaluation.  
- **Full dataset:** Run with `USE_SUBSET = False` and tune batch size / hardware for full-scale experiments.  
- **Threshold tuning:** Experiment with different multipliers (e.g. 2× or 4× std) or percentile-based thresholds.  
- **Additional attack types / datasets:** Extend to more attack classes or other CAN datasets.  
- **Real-time or online setting:** Consider streaming, windowing, and latency for deployment.  
- **Explainability:** Add attention or feature importance to interpret LSTM and autoencoder decisions.

**Challenges:**  
- **Class imbalance:** DoS/Normal (or other) imbalance may require stronger class weighting or resampling.  
- **Threshold sensitivity:** Fixed “mean + 3×std” may not suit all distributions; may need validation-based tuning.  
- **Generalization:** Performance on different vehicles or driving conditions is uncertain; cross-domain validation is important.  
- **Adversarial or novel attacks:** Stage 1 is limited to known attack types; Stage 2 (autoencoder on normal) provides anomaly detection so unknown attacks can be flagged by high reconstruction error, though they may be reported under a known-attack label from Stage 1 unless an explicit “Unknown” bucket is added.

---

## References and Notes

- Implementation: `IDS_new_pipeline.ipynb`.  
- Adjust thresholds, architecture, or data paths as needed for your environment and reporting.
