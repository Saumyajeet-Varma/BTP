# ===============================
# midsem_model_improvised.py
# Car-Hacking IDS — improvised pipeline vs IDS_new_pipeline.ipynb
#
# Problem in original notebook: revamp_predict() returns "Normal" whenever
# autoencoder reconstruction error <= threshold. DoS traffic often uses the
# same IDs/payload shapes as benign traffic (only rate/timing differs), so
# per-message AE loss stays low → DoS collapses to Normal (0 recall).
#
# Fix: (1) rich per-message features including IAT + payload stats
#       (2) sliding windows over time-ordered streams (real temporal context)
#       (3) single 5-class model (Normal, DoS, Fuzzy, Gear, RPM) with
#           stratified split + class weights — no AE gate that erases DoS.
# ===============================

try:
    from google.colab import drive

    drive.mount("/content/drive", force_remount=False)
    _IN_COLAB = True
except ImportError:
    _IN_COLAB = False

import os
import re
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    LSTM,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Input,
    MaxPooling1D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

if _IN_COLAB:
    data_path = "/content/drive/MyDrive/dataset/9) Car-Hacking Dataset"
else:
    data_path = r"9) Car-Hacking Dataset"

USE_SUBSET = True
MAX_NORMAL = 10_000
MAX_PER_ATTACK_FILE = 20_000

SEQ_LEN = 24
BATCH_SIZE = 64
EPOCHS = 60
PATIENCE = 12
VAL_SPLIT = 0.15

print("Device / GPU:", tf.config.list_physical_devices("GPU"))


# (preferred_name, repo_alt_name, attack_label)
ATTACK_FILE_SPECS = [
    ("DoS_dataset.csv", "dos_attack.csv", "DoS"),
    ("Fuzzy_dataset.csv", "fuzzy_attack.csv", "Fuzzy"),
    ("gear_dataset.csv", "gear_spoofing.csv", "Gear"),
    ("RPM_dataset.csv", "rpm_spoofing.csv", "RPM"),
]


def parse_line(line):
    regex = r"Timestamp:\s*(\d+\.?\d*)\s+ID:\s*(\w+)\s+000\s+DLC:\s*(\d+)\s+([\da-fA-F\s]+)"
    match = re.match(regex, line.strip())
    if match:
        timestamp = float(match.group(1))
        can_id = int(match.group(2), 16)
        dlc = int(match.group(3))
        data = [int(byte, 16) for byte in match.group(4).split()]
        data = (data + [0] * 8)[:8]
        return {"Timestamp": timestamp, "CAN_ID": can_id, "DLC": dlc, "DATA": data}
    return None


def load_normal_df(base_path):
    candidates = [
        os.path.join(base_path, "normal_run_data.txt"),
        os.path.join(base_path, "normal_run_data", "normal_run_data.txt"),
    ]
    file_path = None
    for p in candidates:
        if os.path.isfile(p):
            file_path = p
            break
    if file_path is None:
        raise FileNotFoundError("normal_run_data.txt not found under " + repr(base_path))

    rows = []
    with open(file_path, "r") as f:
        for line in f:
            if USE_SUBSET and len(rows) >= MAX_NORMAL:
                break
            p = parse_line(line)
            if p:
                rows.append(p)
    df = pd.DataFrame(rows)
    for i in range(8):
        df["DATA{}".format(i)] = df["DATA"].apply(lambda x, i=i: x[i] if i < len(x) else 0)
    df.drop(columns=["DATA"], inplace=True)
    df["Label"] = "Normal"
    return df


def convert_numeric_columns(df, columns_to_convert):
    for col in columns_to_convert:
        if col == "CAN_ID" or col.startswith("DATA"):
            df[col] = (
                df[col]
                .astype(str)
                .apply(lambda x: int(x, 16) if re.match(r"^[0-9a-fA-F]+$", str(x).strip()) else np.nan)
            )
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(0).astype(int)
    return df


def label_from_flag(flag_series, attack_name):
    labels = []
    for v in flag_series.astype(str).str.strip().str.upper():
        if v == "R":
            labels.append("Normal")
        elif v == "T":
            labels.append(attack_name)
        else:
            labels.append("Normal")
    return labels


def load_attack_df(base_path, primary_name, alt_name, attack_label):
    p1 = os.path.join(base_path, primary_name)
    p2 = os.path.join(base_path, alt_name) if alt_name != primary_name else None
    path = p1 if os.path.isfile(p1) else (p2 if p2 and os.path.isfile(p2) else None)
    if path is None:
        raise FileNotFoundError("Missing attack file: {} / {}".format(primary_name, alt_name))

    column_names = (
        ["Timestamp", "CAN_ID", "DLC"]
        + ["DATA{}".format(i) for i in range(8)]
        + ["Flag"]
    )
    nrows = MAX_PER_ATTACK_FILE if USE_SUBSET else None
    df = pd.read_csv(path, header=None, names=column_names, nrows=nrows)
    cols_to_process = ["CAN_ID", "DLC"] + ["DATA{}".format(i) for i in range(8)]
    df = convert_numeric_columns(df, cols_to_process)
    df["Label"] = label_from_flag(df["Flag"], attack_label)
    return df


def add_engineered_features(df):
    """Timing + payload statistics — DoS differs mainly in IAT / burst density."""
    df = df.sort_values("Timestamp").reset_index(drop=True)
    ts = df["Timestamp"].values.astype(np.float64)
    iat = np.diff(ts, prepend=ts[0])
    iat[0] = 0.0
    iat = np.clip(iat, 0.0, 1.0)
    df["IAT"] = iat

    freq = df["CAN_ID"].value_counts(normalize=True)
    df["CAN_ID_freq"] = df["CAN_ID"].map(freq).fillna(0.0).astype(np.float64)

    data_cols = ["DATA{}".format(i) for i in range(8)]

    def row_entropy(row):
        vals = row.values.astype(float)
        vals = vals[vals > 0]
        if len(vals) == 0:
            return 0.0
        p = vals / vals.sum()
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p + 1e-12)))

    df["byte_entropy"] = df[data_cols].apply(row_entropy, axis=1)
    df["byte_sum"] = df[data_cols].sum(axis=1).astype(np.float64)
    df["byte_range"] = (df[data_cols].max(axis=1) - df[data_cols].min(axis=1)).astype(np.float64)
    df["byte_std"] = df[data_cols].std(axis=1).fillna(0.0).astype(np.float64)
    return df


def make_windows_from_sorted_df(df, base_cols, seq_len):
    """
    Build (N, seq_len, F) windows; label = label at last timestep in window.
    Call only on a single time-ordered stream (one file / session).
    """
    df = add_engineered_features(df)
    feat_cols = base_cols + [
        "IAT",
        "CAN_ID_freq",
        "byte_entropy",
        "byte_sum",
        "byte_range",
        "byte_std",
    ]
    X = df[feat_cols].values.astype(np.float64)
    y_str = df["Label"].values
    if len(X) < seq_len:
        return np.zeros((0, seq_len, len(feat_cols)), np.float32), np.array([], dtype=object)

    n = len(X) - seq_len + 1
    f = X.shape[1]
    Xw = np.zeros((n, seq_len, f), dtype=np.float32)
    yw = np.empty(n, dtype=object)
    for i in range(n):
        Xw[i] = X[i : i + seq_len]
        yw[i] = y_str[i + seq_len - 1]
    return Xw, yw


def build_all_windows(data_path):
    base_features = ["CAN_ID", "DLC"] + ["DATA{}".format(i) for i in range(8)]
    parts = []

    parts.append(load_normal_df(data_path))
    for primary, alt, attack_label in ATTACK_FILE_SPECS:
        parts.append(load_attack_df(data_path, primary, alt, attack_label))

    X_list, y_list = [], []
    for sub in parts:
        if "Flag" in sub.columns:
            sub = sub.drop(columns=["Flag"])
        Xw, yw = make_windows_from_sorted_df(sub, base_features, SEQ_LEN)
        if len(Xw) > 0:
            X_list.append(Xw)
            y_list.append(yw)

    if not X_list:
        raise RuntimeError("No sliding windows built — check dataset paths and SEQ_LEN.")
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    return X_all, y_all


def build_model(seq_len, n_features, num_classes):
    """
    Two-stage deep head (single end-to-end model):
      Stage 1 — Conv1D stack: local temporal patterns along SEQ_LEN.
      Stage 2 — Bi-temporal LSTM + MLP: sequence summary -> 5-class softmax.
    """
    inp = Input(shape=(seq_len, n_features))
    x = Conv1D(64, 3, padding="same", activation="relu")(inp)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.25)(x)
    x = Conv1D(128, 3, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.25)(x)
    x = LSTM(96, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = LSTM(64, return_sequences=False)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.35)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.25)(x)
    out = Dense(num_classes, activation="softmax")(x)
    model = Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if not os.path.isdir(data_path):
    print("ERROR: Dataset folder not found:", repr(data_path))
    print("Set data_path to your Car-Hacking directory (same as IDS_new_pipeline.ipynb).")
else:
    if USE_SUBSET:
        print(
            "Subset: max {:,} normal rows, max {:,} rows per attack CSV".format(
                MAX_NORMAL, MAX_PER_ATTACK_FILE
            )
        )

    X_w, y_str = build_all_windows(data_path)
    print("Windows:", X_w.shape, "labels:", len(y_str))

    le = LabelEncoder()
    y_enc = le.fit_transform(y_str)
    num_classes = len(le.classes_)
    y_cat = to_categorical(y_enc, num_classes=num_classes)

    n_feat = X_w.shape[2]
    X_flat = X_w.reshape(X_w.shape[0], -1)
    scaler = MinMaxScaler()
    X_flat_s = scaler.fit_transform(X_flat)
    X_s = X_flat_s.reshape(X_w.shape[0], SEQ_LEN, n_feat)

    X_train, X_test, y_train, y_test = train_test_split(
        X_s, y_cat, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc
    )

    classes = np.unique(np.argmax(y_train, axis=1))
    cw = compute_class_weight("balanced", classes=classes, y=np.argmax(y_train, axis=1))
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, cw)}
    print("Class weights:", class_weight_dict)

    model = build_model(SEQ_LEN, n_feat, num_classes)
    model.summary()

    cb = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_split=VAL_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=cb,
        verbose=2,
    )

    pred_proba = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    pred_enc = np.argmax(pred_proba, axis=1)
    true_enc = np.argmax(y_test, axis=1)
    pred_labels = le.inverse_transform(pred_enc)
    true_labels = le.inverse_transform(true_enc)

    labels_order = list(le.classes_)
    cm = confusion_matrix(true_labels, pred_labels, labels=labels_order)

    acc = accuracy_score(true_labels, pred_labels)
    prec_macro = precision_score(true_labels, pred_labels, average="macro", zero_division=0)
    rec_macro = recall_score(true_labels, pred_labels, average="macro", zero_division=0)
    f1_macro = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    prec_weighted = precision_score(true_labels, pred_labels, average="weighted", zero_division=0)
    rec_weighted = recall_score(true_labels, pred_labels, average="weighted", zero_division=0)
    f1_weighted = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)

    print("\n========== Overall metrics ==========")
    print("Accuracy:           {:.4f}".format(acc))
    print("Precision (macro): {:.4f}   (weighted): {:.4f}".format(prec_macro, prec_weighted))
    print("Recall (macro):     {:.4f}   (weighted): {:.4f}".format(rec_macro, rec_weighted))
    print("F1-score (macro):   {:.4f}   (weighted): {:.4f}".format(f1_macro, f1_weighted))

    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
        true_labels, pred_labels, labels=labels_order, zero_division=0
    )
    print("\n========== Per-class (precision, recall, F1, support) ==========")
    for i, lab in enumerate(labels_order):
        print(
            "  {:8s}  P={:.4f}  R={:.4f}  F1={:.4f}  n={}".format(
                lab, p_per[i], r_per[i], f1_per[i], int(sup_per[i])
            )
        )

    print("\nConfusion matrix (rows=true, cols=pred):")
    print(labels_order)
    print(cm)

    print("\nClassification report:")
    print(classification_report(true_labels, pred_labels, labels=labels_order, digits=4))

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    cm_path = os.path.join(script_dir, "midsem_model_improvised_confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels_order,
        yticklabels=labels_order,
        ax=ax,
        cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(
        "Midsem improvised IDS — confusion matrix\nAcc={:.4f}  macro P/R/F1={:.3f}/{:.3f}/{:.3f}".format(
            acc, prec_macro, rec_macro, f1_macro
        )
    )
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close(fig)
    print("\nSaved confusion matrix figure:", cm_path)

    print("\n--- Done. ---")
