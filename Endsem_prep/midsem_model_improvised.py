# ===============================
# midsem_model_improvised.py
# Car-Hacking IDS — two-stage hybrid + GAN probes -> zero_day
#
# Stage 1: 5-class classifier — Normal, DoS, Fuzzy, Gear, RPM (known attacks).
# Stage 2: Autoencoder on genuine Normal windows only; scores windows that
#          Stage 1 labels as Normal. High reconstruction error -> zero_day.
#
# GAN: trained on flattened normal windows in [0,1]; generates fake windows.
#      Synthetic eval set (true label zero_day) checks Stage-2 catches them.
#
# Final labels (6): Normal, DoS, Fuzzy, Gear, RPM, zero_day
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
from tensorflow.keras.optimizers import Adam
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
STAGE1_EPOCHS = 55
STAGE1_PATIENCE = 12
STAGE1_VAL_SPLIT = 0.15

AE_EPOCHS = 45
AE_PATIENCE = 8
AE_BATCH = 256

GAN_NOISE_DIM = 48
GAN_EPOCHS = 25
GAN_STEPS_PER_EPOCH = 120
GAN_BATCH = 128
NUM_GAN_TEST_WINDOWS = 1500

STAGE2_NORMAL_PERCENTILE = 98.5

print("Device / GPU:", tf.config.list_physical_devices("GPU"))

ATTACK_FILE_SPECS = [
    ("DoS_dataset.csv", "dos_attack.csv", "DoS"),
    ("Fuzzy_dataset.csv", "fuzzy_attack.csv", "Fuzzy"),
    ("gear_dataset.csv", "gear_spoofing.csv", "Gear"),
    ("RPM_dataset.csv", "rpm_spoofing.csv", "RPM"),
]

FINAL_LABELS = ["Normal", "DoS", "Fuzzy", "Gear", "RPM", "zero_day"]


def _output_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


def plot_confusion_heatmap(y_true, y_pred, labels, title, filename, figsize=(10, 8)):
    """Save confusion matrix heatmap; labels order fixed."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy_score(y_true, y_pred)
    p_m = precision_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
    r_m = recall_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
    f_m = f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
    path = os.path.join(_output_dir(), filename)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(
        "{}\nAcc={:.4f}  macro P/R/F1={:.4f}/{:.4f}/{:.4f}".format(title, acc, p_m, r_m, f_m)
    )
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved:", path)
    return cm, acc, p_m, r_m, f_m


def print_metrics_block(name, y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    p_mac = precision_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
    r_mac = recall_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
    f_mac = f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
    p_w = precision_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels)
    r_w = recall_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels)
    f_w = f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels)
    print("\n========== {} ==========".format(name))
    print("Accuracy:            {:.4f}".format(acc))
    print("Precision (macro):   {:.4f}   (weighted): {:.4f}".format(p_mac, p_w))
    print("Recall (macro):      {:.4f}   (weighted): {:.4f}".format(r_mac, r_w))
    print("F1-score (macro):    {:.4f}   (weighted): {:.4f}".format(f_mac, f_w))
    print(classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0))


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
    parts = [load_normal_df(data_path)]
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
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


def build_stage1_model(seq_len, n_features, num_classes=5):
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
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_autoencoder(flat_dim):
    inp = Input(shape=(flat_dim,))
    x = Dense(256, activation="relu")(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(256, activation="relu")(x)
    out = Dense(flat_dim, activation="sigmoid")(x)
    ae = Model(inp, out)
    ae.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    return ae


def build_gan(flat_dim, noise_dim=GAN_NOISE_DIM):
    """Generator + Discriminator for flat [0,1] windows."""
    gen_in = Input(shape=(noise_dim,))
    g = Dense(256, activation="relu")(gen_in)
    g = BatchNormalization()(g)
    g = Dense(512, activation="relu")(g)
    g = BatchNormalization()(g)
    g = Dense(flat_dim, activation="sigmoid")(g)
    generator = Model(gen_in, g, name="generator")

    disc_in = Input(shape=(flat_dim,))
    d = Dense(256, activation="relu")(disc_in)
    d = Dropout(0.3)(d)
    d = Dense(128, activation="relu")(d)
    d = Dropout(0.2)(d)
    d = Dense(1, activation="sigmoid")(d)
    discriminator = Model(disc_in, d, name="discriminator")
    discriminator.compile(optimizer=Adam(learning_rate=2e-4, beta_1=0.5), loss="binary_crossentropy")

    discriminator.trainable = False
    gan_out = discriminator(generator(gen_in))
    gan = Model(gen_in, gan_out, name="gan")
    gan.compile(optimizer=Adam(learning_rate=2e-4, beta_1=0.5), loss="binary_crossentropy")

    return generator, discriminator, gan


def train_gan(generator, discriminator, gan, X_normal_flat, epochs, steps_per_epoch, batch_size):
    """Train GAN on real normal flattened windows."""
    n = len(X_normal_flat)
    if n < batch_size:
        return
    for _ep in range(epochs):
        for _ in range(steps_per_epoch):
            idx = np.random.randint(0, n, size=batch_size)
            real_x = X_normal_flat[idx].astype(np.float32)
            noise = np.random.normal(0, 1, (batch_size, GAN_NOISE_DIM)).astype(np.float32)
            fake_x = generator.predict(noise, verbose=0)

            discriminator.trainable = True
            discriminator.train_on_batch(real_x, np.ones((batch_size, 1)))
            discriminator.train_on_batch(fake_x, np.zeros((batch_size, 1)))

            discriminator.trainable = False
            noise2 = np.random.normal(0, 1, (batch_size, GAN_NOISE_DIM)).astype(np.float32)
            gan.train_on_batch(noise2, np.ones((batch_size, 1)))


def ae_reconstruction_mse(ae, X_flat):
    recon = ae.predict(X_flat, batch_size=AE_BATCH, verbose=0)
    return np.mean(np.square(X_flat - recon), axis=1)


if not os.path.isdir(data_path):
    print("ERROR: Dataset folder not found:", repr(data_path))
else:
    if USE_SUBSET:
        print(
            "Subset: max {:,} normal rows, max {:,} per attack CSV".format(
                MAX_NORMAL, MAX_PER_ATTACK_FILE
            )
        )

    X_w, y_str = build_all_windows(data_path)
    print("Windows:", X_w.shape)

    le5 = LabelEncoder()
    y_enc = le5.fit_transform(y_str)
    n_feat = X_w.shape[2]
    flat_dim = SEQ_LEN * n_feat

    X_flat = X_w.reshape(len(X_w), flat_dim)
    scaler = MinMaxScaler()
    X_flat_s = scaler.fit_transform(X_flat).astype(np.float32)
    X_s = X_flat_s.reshape(len(X_w), SEQ_LEN, n_feat)

    y_cat5 = to_categorical(y_enc, num_classes=5)

    X_train, X_test, y_train_cat, y_test_cat, y_train_enc, y_test_enc = train_test_split(
        X_s,
        y_cat5,
        y_enc,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_enc,
    )
    n_train = len(X_train)
    X_train_flat = X_train.reshape(n_train, flat_dim)
    X_test_flat = X_test.reshape(len(X_test), flat_dim)

    normal_idx5 = int(le5.transform(["Normal"])[0])
    is_train_normal = y_train_enc == normal_idx5
    X_train_normal_flat = X_train_flat[is_train_normal]
    print("Stage-2 (AE) train normal windows:", len(X_train_normal_flat))

    classes = np.unique(y_train_enc)
    cw = compute_class_weight("balanced", classes=classes, y=y_train_enc)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, cw)}

    # ----- Stage 1 -----
    stage1 = build_stage1_model(SEQ_LEN, n_feat, 5)
    cb1 = [
        EarlyStopping(monitor="val_loss", patience=STAGE1_PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
    ]
    stage1.fit(
        X_train,
        y_train_cat,
        validation_split=STAGE1_VAL_SPLIT,
        epochs=STAGE1_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=cb1,
        verbose=2,
    )

    s1_test_pred = np.argmax(stage1.predict(X_test, batch_size=BATCH_SIZE, verbose=0), axis=1)
    s1_test_true = y_test_enc
    y_s1_true_labels = le5.inverse_transform(s1_test_true)
    y_s1_pred_labels = le5.inverse_transform(s1_test_pred)
    labels5 = list(le5.classes_)
    print_metrics_block("Stage 1 only (5-class: known attacks + Normal)", y_s1_true_labels, y_s1_pred_labels, labels5)
    plot_confusion_heatmap(
        y_s1_true_labels,
        y_s1_pred_labels,
        labels5,
        "Stage 1 — known attacks + Normal",
        "midsem_cm_stage1.png",
    )

    # ----- Autoencoder (Stage 2) on normal only -----
    ae = build_autoencoder(flat_dim)
    ae_cb = EarlyStopping(monitor="val_loss", patience=AE_PATIENCE, restore_best_weights=True)
    ae.fit(
        X_train_normal_flat,
        X_train_normal_flat,
        validation_split=0.1,
        epochs=AE_EPOCHS,
        batch_size=AE_BATCH,
        callbacks=[ae_cb],
        verbose=2,
    )

    train_normal_mse = ae_reconstruction_mse(ae, X_train_normal_flat)
    thresh_base = float(np.percentile(train_normal_mse, STAGE2_NORMAL_PERCENTILE))

    # ----- GAN on normal flat -----
    gen, disc, gan_m = build_gan(flat_dim, GAN_NOISE_DIM)
    train_gan(gen, disc, gan_m, X_train_normal_flat, GAN_EPOCHS, GAN_STEPS_PER_EPOCH, GAN_BATCH)

    noise_eval = np.random.normal(0, 1, (1024, GAN_NOISE_DIM)).astype(np.float32)
    fake_probe = gen.predict(noise_eval, verbose=0)
    gan_mse = ae_reconstruction_mse(ae, fake_probe)
    # Threshold: keep most real normal below T; push typical GAN recon error above T when possible
    threshold = max(thresh_base, float(np.percentile(gan_mse, 8)))
    frac_gan_flagged = float(np.mean(gan_mse > threshold))
    if frac_gan_flagged < 0.5:
        threshold = float(0.5 * (np.percentile(train_normal_mse, 99.5) + np.median(gan_mse)))
        frac_gan_flagged = float(np.mean(gan_mse > threshold))
    print(
        "Stage-2 AE threshold (recon MSE): {:.6f}  (normal p{:.1f}={:.6f}, median GAN MSE={:.6f}, frac GAN flagged={:.2f})".format(
            threshold, STAGE2_NORMAL_PERCENTILE, thresh_base, float(np.median(gan_mse)), frac_gan_flagged
        )
    )

    # ----- GAN test windows (true label zero_day) -----
    noise_test = np.random.normal(0, 1, (NUM_GAN_TEST_WINDOWS, GAN_NOISE_DIM)).astype(np.float32)
    X_gan_flat = gen.predict(noise_test, verbose=0).astype(np.float32)
    X_gan_seq = X_gan_flat.reshape(-1, SEQ_LEN, n_feat)

    X_eval_seq = np.concatenate([X_test, X_gan_seq], axis=0)
    X_eval_flat = np.concatenate([X_test_flat, X_gan_flat], axis=0)
    y_eval_true_str = np.concatenate(
        [le5.inverse_transform(y_test_enc), np.full(NUM_GAN_TEST_WINDOWS, "zero_day", dtype=object)]
    )

    s1_probs_eval = stage1.predict(X_eval_seq, batch_size=BATCH_SIZE, verbose=0)
    s1_eval_pred = np.argmax(s1_probs_eval, axis=1)
    errs_eval = ae_reconstruction_mse(ae, X_eval_flat)

    final_pred = []
    for i in range(len(s1_eval_pred)):
        if s1_eval_pred[i] != normal_idx5:
            final_pred.append(le5.inverse_transform(np.array([s1_eval_pred[i]]))[0])
        else:
            final_pred.append("zero_day" if errs_eval[i] > threshold else "Normal")

    # Stage-2 confusion: only where Stage-1 said Normal; binary Normal vs zero_day truth
    mask_s1_normal = s1_eval_pred == normal_idx5
    y_bin_true = []
    y_bin_pred = []
    for i in range(len(y_eval_true_str)):
        if not mask_s1_normal[i]:
            continue
        t = y_eval_true_str[i]
        if t not in ("Normal", "zero_day"):
            continue
        y_bin_true.append(t)
        y_bin_pred.append("zero_day" if errs_eval[i] > threshold else "Normal")
    labels_bin = ["Normal", "zero_day"]
    if len(y_bin_true) > 0:
        print_metrics_block(
            "Stage 2 only (subset: Stage-1 predicted Normal; truth Normal vs zero_day)",
            y_bin_true,
            y_bin_pred,
            labels_bin,
        )
        plot_confusion_heatmap(
            y_bin_true,
            y_bin_pred,
            labels_bin,
            "Stage 2 — AE on Stage-1 Normal (Normal vs zero_day)",
            "midsem_cm_stage2.png",
            figsize=(7, 6),
        )
    else:
        print("Stage-2 CM skipped (no samples in subset).")

    print_metrics_block("Final hybrid pipeline (6 classes)", y_eval_true_str, final_pred, FINAL_LABELS)
    plot_confusion_heatmap(
        y_eval_true_str,
        final_pred,
        FINAL_LABELS,
        "Final hybrid — Stage1 attacks OR Stage2 zero_day",
        "midsem_cm_final_hybrid.png",
        figsize=(11, 9),
    )

    print("\n--- Pipeline complete: Stage1 -> (if Normal) AE -> zero_day; GAN fakes used as zero_day eval. ---")
