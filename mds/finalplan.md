# IDS Pipeline — Final Report & Plan

This document summarizes the **Intrusion Detection System (IDS)** pipeline implemented in `IDS_new_pipeline.ipynb`, including dataset, preprocessing, pipeline design, model architecture, training, evaluation, rationale, and future scope.

---

## 1. Dataset

- **Source:** Car-Hacking Dataset (CAN bus traffic).
- **Normal traffic:** `normal_run_data/normal_run_data.txt` — text format with parsed fields (Timestamp, CAN_ID, DLC, DATA bytes).
- **Attack traffic (CSV):**  
  - `DoS_dataset.csv`  
  - `Fuzzy_dataset.csv`  
  - `gear_dataset.csv`  
  - `RPM_dataset.csv`  

  Each attack file has columns: `Timestamp`, `CAN_ID`, `DLC`, `DATA0`–`DATA7`, and **`Flag`** (used for per-row labeling).

- **Subset configuration (for manageable runs):**  
  - `USE_SUBSET = True`: cap rows to avoid memory/runtime issues.  
  - `MAX_NORMAL = 10_000` (normal messages).  
  - `MAX_PER_ATTACK_FILE = 20_000` (per attack CSV).  

- **Labels:** Five classes — **Normal**, **DoS**, **Fuzzy**, **Gear**, **RPM**.  
  - Normal file: all rows labeled `"Normal"`.  
  - Attack files: **Flag-based labeling** — `Flag == 'R'` → `"Normal"`, `Flag == 'T'` → attack name (DoS / Fuzzy / Gear / RPM), so each row is labeled by message type, not by file alone.

- **Features used:** `CAN_ID`, `DLC`, `DATA0`, `DATA1`, `DATA2`, `DATA3`, `DATA4`, `DATA5`, `DATA6`, `DATA7` (10 features; `N_FEATURES = 10`).

---

## 2. Data Preprocessing

- **Normal data:**  
  - Parse each line with a regex extracting Timestamp, CAN_ID, DLC, and DATA bytes.  
  - Expand `DATA` into columns `DATA0`–`DATA7` (pad to 8 bytes if needed).  
  - Stop after `MAX_NORMAL` rows when `USE_SUBSET` is True.  
  - Assign `Label = 'Normal'`.

- **Attack data:**  
  - Read CSVs with fixed column names; optionally limit rows via `nrows=MAX_PER_ATTACK_FILE`.  
  - `convert_numeric_columns`: CAN_ID and DATA* as hex → int; DLC and rest numeric; fill NaN with 0.  
  - `label_from_flag(Flag, attack_name)`: R → `'Normal'`, T → attack name (e.g. `'DoS'`), so each row is labeled by message type.

- **Combining:**  
  - `full_df = concat(normal, dos, fuzzy, gear, rpm)`.  
  - Feature matrix `X = full_df[features].values`, labels `y = full_df['Label'].values`.

- **Train/validation split:**  
  - `train_test_split(X, y_cat_full, test_size=0.2, stratify=y_encoded_full, random_state=RANDOM_STATE)`.

- **Scaling:**  
  - `MinMaxScaler` fit on `X_train`, transform both `X_train` and `X_test`.

- **Reshape for LSTM:**  
  - `X_train_lstm` / `X_test_lstm`: shape `(n_samples, 1, N_FEATURES)` for the attack classifier.

---

## 3. Pipeline

The pipeline is **two-stage** and uses a **swapped order** (attack classifier first, then autoencoder):

1. **Stage 01 — Attack-type classifier (run first)**  
   - LSTM model trained only on **attack** samples.  
   - Output: one of **DoS**, **Fuzzy**, **Gear**, **RPM** (4 classes).

2. **Stage 02 — Autoencoder (run second)**  
   - Autoencoder trained only on **normal** traffic.  
   - Reconstruction error on normal training data is used to set a **threshold** (e.g. `mean(train_loss) + 3 * std(train_loss)`).  
   - Used to decide whether a sample is “Normal” (low recon error) or not.

**Prediction logic (`revamp_predict`):**  
- Run **Stage 01** (LSTM) on the sample → get tentative **attack_type** (DoS/Fuzzy/Gear/RPM).  
- Run **Stage 02** (autoencoder) on the sample → compute reconstruction error.  
- If reconstruction error **≤ threshold** → return **"Normal"** (override LSTM).  
- Else → return the **attack_type** from Stage 01.

So: **LSTM first** for every sample; **autoencoder second** to refine “Normal” and reduce false attack labels.

---

## 4. Model Architecture

**Stage 01 — Attack classifier (LSTM)**  
- Input: `(batch, 1, N_FEATURES)`.  
- LSTM(128, return_sequences=False).  
- BatchNormalization.  
- Dropout(0.4).  
- Dense(64, relu).  
- Dropout(0.3).  
- Dense(32, relu).  
- Dense(4, softmax) → DoS, Fuzzy, Gear, RPM.

**Stage 02 — Autoencoder**  
- Input: `(batch, input_dim)` with `input_dim = 10`.  
- Encoder: Dense(64, relu) → BatchNorm → Dense(32, relu) → Dense(16, relu).  
- Decoder: Dense(32, relu) → Dense(64, relu) → Dense(input_dim, linear).  
- Loss: MSE; metric: MAE.

**Threshold:**  
- Computed on **training normal** data: reconstruct with autoencoder, then `threshold = mean(MSE per sample) + 3 * std(MSE per sample)`.

---

## 5. Training Procedure

- **Reproducibility:** `RANDOM_STATE = 42`, `np.random.seed`, `tf.random.set_seed`.

**Stage 01 (attack classifier):**  
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

## 6. Testing and Evaluation

- **Prediction:** For each test sample, `revamp_predict(sample, scaled=True)` runs Stage 01 (LSTM) then Stage 02 (autoencoder) and applies the threshold to decide Normal vs attack type.
- **Metrics:**  
  - `accuracy_score(true_labels, revamp_results)`.  
  - `confusion_matrix` and `classification_report` (precision, recall, f1-score per class).
- **Typical outcome (subset run):** Overall accuracy ~92.78%; high performance on Normal, Gear, RPM; DoS/Fuzzy may need further tuning or more data.  
- **Note:** Per-sample `revamp_predict` is slow for large test sets; batched evaluation (autoencoder on full `X_test`, then LSTM only where needed) can be used for speed.

---

## 7. Why This Model

- **Two-stage design:** Separates (1) “Normal vs attack” and (2) “which attack type,” which fits CAN traffic where normal is abundant and attack types are distinct.
- **Swapped order (LSTM → Autoencoder):**  
  - LSTM sees all samples and gives a tentative attack label.  
  - Autoencoder, trained on normal only, corrects false attack predictions when the sample looks “normal” (low reconstruction error).  
  - This can improve precision on Normal and reduce false positives from the classifier.
- **Flag-based labeling:** Uses the dataset’s `Flag` (R/T) so that mixed attack files are labeled per message, improving supervision quality.
- **Subset option:** Allows fast iteration and debugging without loading the full dataset.

---

## 8. Future Scope and Challenges

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
- **Adversarial or novel attacks:** Model is trained on known attack types; detection of unseen attack patterns would need anomaly or out-of-distribution components.

---

*Generated from the IDS_new_pipeline.ipynb implementation. Adjust thresholds, architecture, or data paths as needed for your environment and reporting.*
