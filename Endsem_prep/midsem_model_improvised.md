# `midsem_model_improvised.py` — pipeline and model stages

This document describes the **end-to-end working** of `midsem_model_improvised.py`: how data flows from raw Car-Hacking files to predictions, how **Stage 1** and **Stage 2** inside the neural network behave, and how that compares to the two-stage design in `codes/IDS_new_pipeline.ipynb`.

---

## 1. Purpose

The script trains an **intrusion detection** model on the **Car-Hacking** dataset: five classes — **Normal**, **DoS**, **Fuzzy**, **Gear**, **RPM**. It is an improvised alternative to the notebook pipeline that had **DoS recall collapsed to zero** because an autoencoder gate treated many DoS frames as “normal-looking” in payload space.

Here, **one Keras model** maps a **fixed-length CAN window** directly to a **5-way softmax**. There is **no** separate autoencoder overriding the class label at inference time.

---

## 2. High-level pipeline (what runs, in order)

The script is structured as a **linear pipeline** after optional Colab `drive.mount`:

| Step | What happens |
|------|----------------|
| **A. Paths & seeds** | `data_path` (Colab Drive or local `9) Car-Hacking Dataset`), `RANDOM_STATE`, TensorFlow seed. |
| **B. Load streams** | Normal traffic from `normal_run_data.txt` (with row cap). Each attack from a CSV (`DoS_dataset.csv` or `dos_attack.csv`, etc.). |
| **C. Row labels** | Attack CSVs use `Flag`: `R` → Normal segment inside that recording, `T` → attack type for that file. |
| **D. Feature engineering** | Per stream, sort by `Timestamp`, then add derived columns (IAT, ID frequency, byte statistics). |
| **E. Sliding windows** | For each stream alone: build tensors of shape `(N, SEQ_LEN, n_features)`; label = class at the **last** timestep of the window. |
| **F. Concatenate** | All windows from all streams are stacked into one dataset. |
| **G. Encode labels** | `LabelEncoder` → integers → `to_categorical` one-hot for Keras. |
| **H. Scale** | `MinMaxScaler` on flattened windows, then reshape back to `(N, SEQ_LEN, n_features)`. |
| **I. Split** | Stratified `train_test_split` (80% train, 20% test) on window labels. |
| **J. Class weights** | `compute_class_weight('balanced')` on training labels to reduce bias toward Normal. |
| **K. Build & train model** | Two-stage CNN–RNN architecture (details below), Adam, categorical cross-entropy, early stopping + LR schedule. |
| **L. Evaluate** | Accuracy; macro and weighted precision, recall, F1; per-class P/R/F1/support; confusion matrix **printed** and **saved as a heatmap PNG**. |

Output figure path (same folder as the script, or current working directory if `__file__` is unavailable):

`midsem_model_improvised_confusion_matrix.png`

---

## 3. Data loading (streams, not one shuffled bag of rows)

**Normal:** lines are parsed with a regex into `Timestamp`, `CAN_ID`, `DLC`, eight data bytes; `DATA0`…`DATA7` are expanded; every row is labeled `Normal`. A subset cap `MAX_NORMAL` applies when `USE_SUBSET` is true.

**Attacks:** each file is read with fixed column names; hex fields are converted to integers; `label_from_flag` assigns `Normal` or the file’s attack name from `Flag`.

**Why per-file windows:** Windows are built **inside each dataframe** after sorting by `Timestamp`. That way a window is a short **contiguous slice of one recording**, not an arbitrary mix of unrelated timestamps from different files.

---

## 4. Feature engineering (`add_engineered_features`)

Base features match the notebook spirit: `CAN_ID`, `DLC`, `DATA0`…`DATA7`.

**Added columns:**

- **`IAT` (inter-arrival time):** difference between consecutive timestamps, clipped to `[0, 1]` seconds. Captures **burstiness** (important for DoS).
- **`CAN_ID_freq`:** relative frequency of that ID inside **this** stream — common IDs get higher values.
- **`byte_entropy`:** Shannon entropy of non-zero byte values in the payload — fuzzy / random payloads often differ from steady operational payloads.
- **`byte_sum`, `byte_range`, `byte_std`:** simple payload summaries.

These are computed **after** sorting by time so IAT is meaningful.

---

## 5. Sliding windows (`make_windows_from_sorted_df`)

- Input: one sorted stream + base column list + `SEQ_LEN` (default 24).
- For each start index `i`, the window is rows `i … i+SEQ_LEN-1`, shape `(SEQ_LEN, n_features)`.
- **Label:** the label of row `i+SEQ_LEN-1` (last frame in the window). That ties the prediction to “what is happening **now** given the recent context.”

Output: `X_w` of shape `(num_windows, SEQ_LEN, n_features)` and parallel string labels.

---

## 6. Stage 1 and Stage 2 **inside this script’s model**

The improvised script uses **one `Model` object**, but it is built as **two conceptual stages** (implemented in `build_model`). This is the right place to use the words “Stage 1” and “Stage 2” for **this** codebase.

### Stage 1 — Convolutional temporal front-end

**Layers:** `Conv1D(64) → BatchNorm → MaxPool1D → Dropout → Conv1D(128) → BatchNorm → MaxPool1D → Dropout`.

**Input tensor shape:** `(batch, SEQ_LEN, n_features)` — think of `SEQ_LEN` as time steps and `n_features` as channels per step (like a multivariate time series).

**What it does:**

- Each `Conv1D` slides **small temporal kernels** (size 3) along the sequence. At each position it mixes **neighbouring timesteps** and **all feature channels** into new feature maps.
- **Batch normalization** stabilizes scale across the batch; **max pooling** downsamples time so higher layers see **wider receptive fields** with fewer steps.
- **Dropout** reduces overfitting.

**Intuition:** Stage 1 learns **local motifs** in the CAN window — e.g. short bursts of IDs, rapid changes in IAT, local payload patterns — without yet collapsing the whole window to a single vector.

**Tensor flow:** `(B, T, F) → (B, T', C128)` after pools (length `T'` is shorter than `T`).

### Stage 2 — Recurrent sequence model + classifier head

**Layers:** `LSTM(96, return_sequences=True) → Dropout → LSTM(64, return_sequences=False) → BatchNorm → Dropout → Dense(128, relu) → Dropout → Dense(5, softmax)`.

**What it does:**

- The first **LSTM** reads the downsampled sequence **in order** and keeps a hidden state that summarizes “what has happened so far” at each step; `return_sequences=True` means it still outputs a vector per timestep for the next LSTM.
- The second **LSTM** aggregates that into **one hidden vector for the entire window** (`return_sequences=False`). That vector is a **global summary** of the pattern after Stage 1’s local filtering.
- The **Dense** layers form an MLP that maps that summary to **five logits**; **softmax** turns logits into **class probabilities** that sum to 1.

**Intuition:** Stage 2 answers “given the temporal evolution after local filtering, which of the five classes is this window?” — including DoS when **timing/context** in the window differs from benign stretches.

**Training signal:** Categorical cross-entropy between one-hot true labels and softmax predictions, with **class weights** so rare classes are not ignored.

---

## 7. How this differs from **Stage 1 / Stage 2** in `IDS_new_pipeline.ipynb`

The **notebook** uses a **different** two-stage *system* (two separate models + a rule):

| Notebook stage | Role |
|------------------|------|
| **Stage 1** | An **attack-type classifier** (LSTM + Dense) trained **only on attack rows**, among four attacks. Input was effectively `(batch, 1, n_features)` — almost **no sequence**. |
| **Stage 2** | An **autoencoder** trained on **Normal** only; reconstruction error defines a threshold. |
| **Inference rule** | Always run Stage 1, but if AE error ≤ threshold → output **Normal**, else output Stage 1’s attack class. |

Because DoS often **reconstructs like Normal** under MSE on per-message vectors, Stage 2 **forced** Normal and **erased** DoS regardless of Stage 1.

**This script:** no autoencoder gate; temporal **windows** + richer features + **joint 5-class training** fix that failure mode.

---

## 8. Training configuration (knobs)

- `SEQ_LEN`, `BATCH_SIZE`, `EPOCHS`, `PATIENCE`, `VAL_SPLIT` — see top of the `.py` file.
- `USE_SUBSET`, `MAX_NORMAL`, `MAX_PER_ATTACK_FILE` — control dataset size for fast runs.

---

## 9. Evaluation outputs (metrics + chart)

After `model.predict` on the held-out test windows, the script:

1. Prints **accuracy**.
2. Prints **precision, recall, F1** with **`macro`** and **`weighted`** averaging (sklearn definitions: macro = unweighted mean over classes; weighted = support-weighted mean).
3. Prints **per-class** precision, recall, F1, and support.
4. Prints the numeric **confusion matrix** (rows = true class, columns = predicted class).
5. Prints sklearn’s **classification_report**.
6. Saves **`midsem_model_improvised_confusion_matrix.png`**: a **seaborn heatmap** of the confusion matrix for quick visual inspection of which classes are confused with which.

---

## 10. How to run

- **Colab:** mount Drive, set `data_path`, run the script.
- **Local:** place the Car-Hacking folder where `data_path` points, install `tensorflow`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, then `python midsem_model_improvised.py`.

If the dataset path is wrong, the script prints an error and exits before training.
