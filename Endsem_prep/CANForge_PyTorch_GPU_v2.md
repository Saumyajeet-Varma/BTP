# CANForge_PyTorch_GPU_v2.ipynb — Documentation

This document explains the **CANForge** intrusion-detection notebook: data flow, models, training, and how the **hybrid** classifier + AAE system works.

---

## 1. Purpose

The notebook implements a **Car-Hacking CAN bus** pipeline in **PyTorch** (GPU when available):

- **CANForge** — multi-class classifier: Normal, DoS, Fuzzy, Gear, RPM (known attack families).
- **AAE v2** — adversarial autoencoder trained on **normal** traffic only, used to score **anomalies** (proxy for unseen / zero-day style attacks).
- **Hybrid rule** — combine supervised predictions with AAE scores to label **Known** attacks, **Zero-Day** (anomalous but predicted Normal by the classifier), or **Normal**.

---

## 2. Notebook Structure (Cell Map)

| Section | What it does |
|--------|----------------|
| Title | Describes multi-scale residual CNN-BiLSTM + SE attention + AAE |
| Imports & GPU | Libraries, seeds, `device = cuda/cpu` |
| Data path | `USE_SUBSET`, caps on rows per file |
| Load normal | Parse `normal_run_data.txt` with regex |
| Load attacks | CSVs with `Flag` (R/T) → labels |
| Feature engineering | 16 features, concat `full_df` |
| Preprocessing | `LabelEncoder`, stratified split, `MinMaxScaler` |
| DataLoaders | Tensors shape `(N, 1, 16)` for `Conv1d` |
| **CANForge** | `SEBlock` + `CANForge` module |
| Training | Class weights, Adam, plateau LR, early stopping |
| Evaluation | Accuracy, weighted metrics, `classification_report` |
| Visualization | Confusion matrix, plots |
| Ablation | A1–A3 stripped models vs full CANForge (A4) |
| AAE v2 | Encoder, Decoder, Discriminator, training loop |
| Zero-day eval | Scores, threshold, ROC, per-attack detection |
| AAE plots | Curves, histograms |
| Score diagnostics | Per-class percentiles, Normal vs DoS comparison |
| Hybrid | CANForge + AAE decision table |
| Final report | Text summary (hardware, metrics, times) |

---

## 3. Data Loading

### 3.1 Normal log (`normal_run_data.txt`)

- **`parse_line`** matches lines like: `Timestamp: … ID: … DLC: …` plus hex payload bytes.
- Fields: timestamp, CAN ID (hex → int), DLC, 8 data bytes (zero-padded).
- Builds `df_normal` with columns `DATA0`…`DATA7`, label **Normal**.

### 3.2 Attack CSVs

Files: `dos_attack.csv`, `fuzzy_attack.csv`, `gear_spoofing.csv`, `rpm_spoofing.csv`.

- Columns include a **`Flag`**: **`R`** → treated as **Normal** (reference), **`T`** → that file’s attack name.
- Hex fields (`CAN_ID`, `DATA*`) converted to integers.

This yields **mixed** normal/attack rows inside attack files, matching the dataset’s labeling scheme.

### 3.3 Combine and sort

- Concatenate normal + all attack frames, drop `Flag` after use.
- **Sort by `Timestamp`** so time-based features are meaningful.

---

## 4. Feature Engineering (16 Features)

**Base (10):** `CAN_ID`, `DLC`, `DATA0`…`DATA7`

**Derived (6):**

| Feature | Meaning |
|---------|--------|
| `IAT` | Inter-arrival time (`diff` of timestamp), first row 0, **clipped at 1.0** |
| `CAN_ID_freq` | Relative frequency of this CAN ID in the **combined** dataframe |
| `byte_entropy` | Shannon entropy (log2) over **non-zero** payload bytes in the row |
| `byte_sum` | Sum of 8 DATA bytes |
| `byte_range` | max − min of DATA bytes |
| `byte_std` | Standard deviation across DATA bytes |

These capture timing, ID popularity, and payload statistics useful for both classical attacks and noise-like traffic.

---

## 5. Preprocessing

- **`X`**: matrix `(rows, 16)`.
- **`LabelEncoder`**: string labels → `0 … num_classes-1`.
- **`train_test_split`**: 80% train / 20% test, **stratified** on encoded labels.
- **`MinMaxScaler`**: fit on train, transform train and test → values in **[0, 1]** (important for the AAE decoder’s **Sigmoid** output).

### Tensor shapes for CANForge

- **`unsqueeze(1)`** → shape **`(N, 1, 16)`** = batch × **channels** × **sequence length** for **`Conv1d`**.

`DataLoader` uses **`pin_memory=True`** when moving batches to GPU.

---

## 6. Model: CANForge (`CANForge` class)

### 6.1 `SEBlock` (Squeeze-and-Excitation)

- Input: `(batch, channels, seq_len)`.
- **Squeeze**: average over `seq_len` → one value per channel.
- **Excitation**: small MLP → **Sigmoid** gates.
- **Scale**: multiply the feature map by gates (channel-wise attention).

### 6.2 Multi-scale CNN

Three **`Conv1d`** branches on the same input, kernel sizes **1, 3, 5** (each → 32 channels), concatenated → **96 channels**.

### 6.3 Residual block

- `Conv1d(96 → 96)` + **skip connection** from the 96-ch tensor before that conv, then ReLU and dropout.

### 6.4 BiLSTM stack

- Permute to **`(batch, seq_len=16, features=96)`** so each “time step” is one feature index along the 16-length sequence.
- Two **bidirectional LSTM** layers (with dropout between stacked LSTMs where applicable), **BatchNorm** on the channel dimension (via permute), and a **residual add** between first and second LSTM outputs.

### 6.5 Classification head

- **Global average pooling** over sequence length → vector size **128**.
- MLP → **`num_classes`** logits.

---

## 7. Training (CANForge)

- **`compute_class_weight('balanced', …)`** passed into **`CrossEntropyLoss(weight=…)`** to handle class imbalance.
- **Adam**, learning rate scheduler **`ReduceLROnPlateau`** on validation loss.
- Train/validation split taken **from the training portion** of the 80% split (e.g. 85% / 15%) for early stopping.
- Best **`state_dict`** by validation loss is restored after training.

---

## 8. Evaluation and Ablation

- Test loop: softmax probabilities, argmax predictions.
- Reports **accuracy**, weighted precision/recall/F1, per-class **`classification_report`**, confusion matrix.
- **Ablation** trains smaller variants (single-scale CNN+BiLSTM, multi-scale without SE/residual, etc.) with a helper for fewer epochs, then compares to the full CANForge result already trained.

---

## 9. AAE v2 (Adversarial Autoencoder)

### 9.1 Architecture

- **Encoder**: `Linear` stack with **LayerNorm**, **LeakyReLU**, light dropout → **`LATENT_DIM`** (e.g. 16).
- **Decoder**: mirror → **`n_features`** with final **Sigmoid** (matches MinMax inputs).
- **LatentDiscriminator**: **spectral norm** linear layers; scores whether a latent vector looks like **Gaussian** (“real”) vs **encoder output** (“fake”).

### 9.2 Training data

Only **Normal** rows from the **training** split (scaled) are used to learn the normal manifold.

### 9.3 Per-batch steps (simplified)

1. **Denoising AE**: add small Gaussian noise to inputs, encode noisy → decode, **`SmoothL1Loss`** vs **clean** input; update encoder+decoder.
2. **Discriminator**: real `z ~ N(0,I)` vs `enc(x)` detached; BCE with soft labels (e.g. 0.9 / 0.1).
3. **Generator (encoder)**: encourage `enc(x)` to fool the discriminator.

Validation MSE on normal val split drives **ReduceLROnPlateau** and **early stopping**; best weights are reloaded.

---

## 10. Anomaly Scores and Threshold

On the test set (and calibration):

- **Reconstruction error** per sample: mean squared error between `x` and `dec(enc(x))`.
- **Latent norm** `||z||`.
- Normalize deviations using **mean/std from a normal validation** set (z-score style).
- **Combined score** = `(1 - LATENT_WEIGHT) * recon_zscore + LATENT_WEIGHT * latent_zscore`.
- **Threshold**: e.g. **97th percentile** of combined scores on the appropriate normal validation set (see notebook for exact definition used in your run).

Binary metrics and **ROC AUC** treat “attack” as any class ≠ Normal. Per-attack **detection rate** = fraction of that attack’s test points with score above threshold.

---

## 11. Hybrid Decision System

For each test sample `i`:

1. If **CANForge** predicts **not Normal** → output **`Known:<predicted_class>`**.
2. Else if **CANForge** predicts **Normal** but **`anomaly_score > threshold`** → **`ZeroDay`**.
3. Else → **`Normal`**.

This is how the notebook simulates **catching traffic the classifier would miss** using the AAE, while still using the classifier for **known** attack types when it is confident.

---

## 12. How to Run

1. Place the **Car-Hacking** dataset under the path set in **`data_path`** (default: `9) Car-Hacking Dataset` next to the notebook, or adjust the path).
2. Run cells **top to bottom** on a machine with PyTorch; GPU optional but recommended.
3. Tune **`USE_SUBSET`**, **`MAX_NORMAL`**, **`MAX_PER_ATTACK_FILE`**, epochs, and AAE hyperparameters if memory or runtime is tight.

---

## 13. Practical Notes

- **DoS** traffic often reconstructs like normal under a pure AE; your diagnostics may show **low** DoS anomaly scores. That is a **data/model limitation**, not necessarily a bug.
- **Hybrid** metrics trade **precision** vs **recall**: aggressive thresholds catch more “zero-days” but may flag more benign traffic.
- Keep **`MinMaxScaler`** consistent between classifier tensors `(N,1,16)` and AAE flat inputs `(N,16)` as in the notebook.

---

*Generated as companion documentation for `Endsem_prep/CANForge_PyTorch_GPU_v2.ipynb`.*
