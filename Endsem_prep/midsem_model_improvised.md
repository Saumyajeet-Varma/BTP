# Midsem model (improvised) — Car-Hacking IDS

This note accompanies `midsem_model_improvised.py` in the same folder. It is written to align with the style and data layout of `codes/IDS_new_pipeline.ipynb` (Colab drive mount, Car-Hacking CSVs, Keras/TensorFlow), but it fixes a structural flaw that drove **DoS recall to zero** in the notebook’s evaluation.

## Why DoS failed in `IDS_new_pipeline.ipynb`

The notebook pipeline does two things:

1. **Attack-type head**: an LSTM on input shape `(batch, 1, 10)` — effectively **one timestep**, so the recurrent layer adds almost no temporal information.
2. **Autoencoder gate**: `revamp_predict` returns `"Normal"` whenever reconstruction MSE is **below** a threshold learned on normal traffic.

DoS on CAN buses often **reuses legitimate IDs and payload shapes**; the attack is mainly **timing / bus load**. Per-message vectors then look “normal” to the AE, MSE stays low, and the function returns **Normal** before the attack label matters — hence **DoS precision/recall 0** in the reported `classification_report`.

## What this improvised script changes

| Aspect | Notebook (`IDS_new_pipeline`) | Improvised script |
|--------|-------------------------------|---------------------|
| Temporal context | Reshape to `(N, 1, F)` | **Sliding windows** `(N, SEQ_LEN, F)` per file/stream, time-sorted |
| Features | `CAN_ID`, `DLC`, `DATA0`–`DATA7` | Same base columns plus **IAT**, **CAN_ID frequency**, **byte entropy / sum / range / std** |
| Decision logic | AE gate can force **Normal** | **Single 5-class softmax** (Normal + four attacks); no AE override |
| Class imbalance | Balanced weights on attack-only head | **`compute_class_weight('balanced')`** on the full 5-class training set |

Windows are built **separately** for normal text and each attack CSV (sorted by `Timestamp` inside that stream), so labels do not leak across unrelated recordings the way a single global sort might.

## Model architecture (Keras)

- `Conv1D` → `MaxPool` → `Conv1D` → `MaxPool` → stacked **LSTM** → `Dense` → softmax over five classes.
- Training: Adam, categorical cross-entropy, early stopping, learning-rate reduction on `val_loss`.

## Configuration knobs

- `data_path`: Colab Drive path or local `9) Car-Hacking Dataset` (same idea as the notebook).
- `USE_SUBSET`, `MAX_NORMAL`, `MAX_PER_ATTACK_FILE`: match the notebook’s subset behaviour.
- `SEQ_LEN` (default 24): window length; increase if you have enough rows per stream and want longer context.

## Attack CSV file names

The loader tries the **notebook names first** (`DoS_dataset.csv`, …), then the **alternate names** used elsewhere in this repo (`dos_attack.csv`, …).

## How to run

- **Colab**: ensure the dataset path matches `data_path`, then run the script as a single cell or `!python midsem_model_improvised.py`.
- **Local**: point `data_path` at your Car-Hacking folder (or place the dataset next to the script under `9) Car-Hacking Dataset`).

You should see a **non-zero DoS recall** in the printed `classification_report`, assuming the subset still contains enough `Flag == T` DoS rows inside each attack file.

## Optional next steps (not implemented here)

- If you want to **keep** an autoencoder for novelty detection, use it as a **second stage only for “unknown”** (e.g. high AE error **and** low softmax confidence), not as a filter that can relabel traffic as Normal.
- Per-class **Focal Loss** (as in `endsem_new_pipeline.py` PyTorch code) can further help rare classes; here, class weights plus temporal features are usually enough for DoS on this dataset.
