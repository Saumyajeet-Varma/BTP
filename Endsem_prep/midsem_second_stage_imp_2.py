# ===============================
# midsem_second_stage_imp_2.py
# Extends midsem_second_stage_imp.py with:
#   - Val threshold + Stage-1 gate tuned together: maximize F-beta (beta<1, default 0.5)
#     subject to minimum recall on GAN positives (Normal vs zero_day, S1=Normal only).
#   - Two-criterion Stage-2 decision on S1=Normal windows: zero_day iff
#       (combined_score > T) AND (p(Normal|Stage1) <= CAP)
#     so very confident "Normal" softmax suppresses AE spikes (fewer Normal→zero_day FPs).
#
# Outputs: midsecond2_cm_stage1.png, midsecond2_cm_stage2.png, midsecond2_cm_final_hybrid.png
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
    fbeta_score,
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

AE_LATENT_DIM = 40
AE_EPOCHS = 50
AE_PATIENCE = 10
AE_BATCH = 256

GAN_NOISE_DIM = 48
GAN_EPOCHS = 28
GAN_STEPS_PER_EPOCH = 130
GAN_BATCH = 128
NUM_GAN_VAL_PROBES = 2500
NUM_GAN_TEST_WINDOWS = 1500

THRESHOLD_GRID_POINTS = 400
# Val tuning: favor precision on zero_day vs false alarms (beta<1 weights precision more)
FBETA_TUNING = 0.5
# Minimum recall on synthetic zero_day (GAN) positives on val; if infeasible, relax automatically
MIN_RECALL_ZERO_DAY_VAL = 0.70
# Search CAP in [CAP_LO, CAP_HI]; p(Normal) must be <= CAP to allow zero_day
CAP_GRID_POINTS = 14

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
        cmap="Greens",
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
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_encoder_decoder_ae(flat_dim, latent_dim=AE_LATENT_DIM):
    """Bottleneck AE; encoder exposes latent z for anomaly scoring."""
    inp = Input(shape=(flat_dim,))
    x = Dense(256, activation="relu")(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.12)(x)
    x = Dense(128, activation="relu")(x)
    z = Dense(latent_dim, activation="relu", name="latent")(x)
    x = Dense(128, activation="relu")(z)
    x = BatchNormalization()(x)
    x = Dropout(0.12)(x)
    x = Dense(256, activation="relu")(x)
    out = Dense(flat_dim, activation="sigmoid")(x)
    ae = Model(inp, out)
    encoder = Model(inp, z, name="encoder")
    ae.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    return ae, encoder


def build_gan(flat_dim, noise_dim=GAN_NOISE_DIM):
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


def compute_train_normal_stats(encoder, ae, X_normal_flat, batch_size):
    """Mean/std of MSE and ||z|| on train-normal for z-scoring."""
    n = len(X_normal_flat)
    mses, norms = [], []
    for i in range(0, n, batch_size):
        chunk = X_normal_flat[i : i + batch_size].astype(np.float32)
        z = encoder.predict(chunk, verbose=0)
        recon = ae.predict(chunk, verbose=0)
        mses.append(np.mean(np.square(chunk - recon), axis=1))
        norms.append(np.linalg.norm(z, axis=1))
    mse_all = np.concatenate(mses)
    lat_all = np.concatenate(norms)
    m_mu, m_std = float(mse_all.mean()), float(mse_all.std() + 1e-8)
    l_mu, l_std = float(lat_all.mean()), float(lat_all.std() + 1e-8)
    return m_mu, m_std, l_mu, l_std


def combined_anomaly_score(encoder, ae, X_flat, m_mu, m_std, l_mu, l_std, batch_size):
    """max(|z_MSE|, |z_lat|) — OOD often spikes one or both."""
    n = len(X_flat)
    scores = []
    for i in range(0, n, batch_size):
        chunk = X_flat[i : i + batch_size].astype(np.float32)
        z = encoder.predict(chunk, verbose=0)
        recon = ae.predict(chunk, verbose=0)
        mse = np.mean(np.square(chunk - recon), axis=1)
        lat = np.linalg.norm(z, axis=1)
        zs_m = np.abs(mse - m_mu) / m_std
        zs_l = np.abs(lat - l_mu) / l_std
        scores.append(np.maximum(zs_m, zs_l))
    return np.concatenate(scores)


def tune_stage2_threshold_fbeta_gate(
    stage1,
    encoder,
    ae,
    X_val_seq,
    X_val_flat,
    y_val_enc,
    gen,
    normal_idx5,
    m_mu,
    m_std,
    l_mu,
    l_std,
):
    """
    Val: true Normal + GAN, only S1=Normal. Binary y: 0=Normal, 1=zero_day (GAN).
    Jointly pick (T, CAP) maximizing F_beta (beta=FBETA_TUNING) with recall on positives
    >= MIN_RECALL_ZERO_DAY_VAL; if no pair satisfies recall, best unconstrained F_beta.
    pred = 1 iff (score > T) & (p_normal <= CAP).
    """
    p_val = stage1.predict(X_val_seq, batch_size=BATCH_SIZE, verbose=0)
    s1 = np.argmax(p_val, axis=1)
    pnorm_val = p_val[:, normal_idx5].astype(np.float64)
    sc_norm = combined_anomaly_score(encoder, ae, X_val_flat, m_mu, m_std, l_mu, l_std, AE_BATCH)

    mask_n = (y_val_enc == normal_idx5) & (s1 == normal_idx5)
    scores_neg = sc_norm[mask_n]
    pnorm_neg = pnorm_val[mask_n]
    if len(scores_neg) < 50:
        m2 = s1 == normal_idx5
        scores_neg = sc_norm[m2]
        pnorm_neg = pnorm_val[m2]

    noise_v = np.random.normal(0, 1, (NUM_GAN_VAL_PROBES, GAN_NOISE_DIM)).astype(np.float32)
    X_gan_v = gen.predict(noise_v, verbose=0).astype(np.float32)
    X_gan_seq = X_gan_v.reshape(-1, SEQ_LEN, X_val_seq.shape[2])
    p_g = stage1.predict(X_gan_seq, batch_size=BATCH_SIZE, verbose=0)
    s1_g = np.argmax(p_g, axis=1)
    pnorm_g = p_g[:, normal_idx5].astype(np.float64)
    sc_gan = combined_anomaly_score(encoder, ae, X_gan_v, m_mu, m_std, l_mu, l_std, AE_BATCH)
    mask_g = s1_g == normal_idx5
    scores_pos = sc_gan[mask_g]
    pnorm_pos = pnorm_g[mask_g]
    if len(scores_pos) < 50:
        scores_pos = sc_gan
        pnorm_pos = pnorm_g

    y_bin = np.concatenate([np.zeros(len(scores_neg), dtype=np.int32), np.ones(len(scores_pos), dtype=np.int32)])
    scores_all = np.concatenate([scores_neg, scores_pos])
    pnorm_all = np.concatenate([pnorm_neg, pnorm_pos])

    lo = float(np.min(scores_all))
    hi = float(np.max(scores_all))
    if hi <= lo:
        t0 = 0.5 * (lo + hi)
        cap0 = 0.99
        print("Stage-2 (degenerate scores): T={:.6f} CAP={:.4f} (fallback)".format(t0, cap0))
        return t0, cap0, 0.0, 0.0

    t_candidates = np.linspace(lo, hi, THRESHOLD_GRID_POINTS)
    cap_lo, cap_hi = 0.82, 0.995
    cap_candidates = np.linspace(cap_lo, cap_hi, CAP_GRID_POINTS)

    def metrics_for(pred):
        rec_p = recall_score(y_bin, pred, pos_label=1, average="binary", zero_division=0)
        fb = fbeta_score(
            y_bin,
            pred,
            beta=FBETA_TUNING,
            pos_label=1,
            average="binary",
            zero_division=0,
        )
        pr = precision_score(y_bin, pred, pos_label=1, average="binary", zero_division=0)
        return rec_p, fb, pr

    best_constrained = None
    best_unconstrained = None

    for t in t_candidates:
        for cap in cap_candidates:
            pred = ((scores_all > t) & (pnorm_all <= cap)).astype(np.int32)
            rec_p, fb, pr = metrics_for(pred)
            key = (fb, pr, rec_p)
            if best_unconstrained is None or key > best_unconstrained[0]:
                best_unconstrained = (key, float(t), float(cap), rec_p, fb, pr)
            if rec_p >= MIN_RECALL_ZERO_DAY_VAL:
                if best_constrained is None or key > best_constrained[0]:
                    best_constrained = (key, float(t), float(cap), rec_p, fb, pr)

    if best_constrained is not None:
        _, best_t, best_cap, rec_p, fb, pr = best_constrained
        mode = "constrained"
    else:
        _, best_t, best_cap, rec_p, fb, pr = best_unconstrained
        mode = "unconstrained (min recall {:.2f} not met on val)".format(MIN_RECALL_ZERO_DAY_VAL)

    f1_bin = f1_score(y_bin, ((scores_all > best_t) & (pnorm_all <= best_cap)).astype(np.int32), zero_division=0)
    print(
        "Stage-2 val tune (F{:.1f} + p(Normal) gate): T={:.6f} CAP={:.4f}  mode={}  "
        "val P/R/F{:.1f}/F1={:.4f}/{:.4f}/{:.4f}/{:.4f}  |neg|={} |pos|={}".format(
            FBETA_TUNING,
            best_t,
            best_cap,
            mode,
            FBETA_TUNING,
            pr,
            rec_p,
            fb,
            f1_bin,
            len(scores_neg),
            len(scores_pos),
        )
    )
    return best_t, best_cap, fb, f1_bin


if not os.path.isdir(data_path):
    print("ERROR: Dataset folder not found:", repr(data_path))
else:
    if USE_SUBSET:
        print("Subset: max {:,} normal, max {:,} per attack CSV".format(MAX_NORMAL, MAX_PER_ATTACK_FILE))

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

    X_train_big, X_test, y_big_enc, y_test_enc = train_test_split(
        X_s, y_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc
    )
    X_fit, X_val, y_fit_enc, y_val_enc = train_test_split(
        X_train_big,
        y_big_enc,
        test_size=0.1875,
        random_state=RANDOM_STATE,
        stratify=y_big_enc,
    )

    X_fit_flat = X_fit.reshape(len(X_fit), flat_dim)
    X_val_flat = X_val.reshape(len(X_val), flat_dim)
    X_test_flat = X_test.reshape(len(X_test), flat_dim)

    y_fit_cat = to_categorical(y_fit_enc, num_classes=5)
    normal_idx5 = int(le5.transform(["Normal"])[0])
    is_fit_normal = y_fit_enc == normal_idx5
    X_fit_normal_flat = X_fit_flat[is_fit_normal]
    print("AE fit normal windows:", len(X_fit_normal_flat))

    classes = np.unique(y_fit_enc)
    cw = compute_class_weight("balanced", classes=classes, y=y_fit_enc)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, cw)}

    stage1 = build_stage1_model(SEQ_LEN, n_feat, 5)
    cb1 = [
        EarlyStopping(monitor="val_loss", patience=STAGE1_PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
    ]
    stage1.fit(
        X_fit,
        y_fit_cat,
        validation_split=STAGE1_VAL_SPLIT,
        epochs=STAGE1_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=cb1,
        verbose=2,
    )

    s1_test_pred = np.argmax(stage1.predict(X_test, batch_size=BATCH_SIZE, verbose=0), axis=1)
    y_s1_true = le5.inverse_transform(y_test_enc)
    y_s1_pred = le5.inverse_transform(s1_test_pred)
    labels5 = list(le5.classes_)
    print_metrics_block("Stage 1 only (5-class)", y_s1_true, y_s1_pred, labels5)
    plot_confusion_heatmap(y_s1_true, y_s1_pred, labels5, "Stage 1 (imp_2 pipeline)", "midsecond2_cm_stage1.png")

    ae, encoder = build_encoder_decoder_ae(flat_dim, AE_LATENT_DIM)
    ae_cb = EarlyStopping(monitor="val_loss", patience=AE_PATIENCE, restore_best_weights=True)
    ae.fit(
        X_fit_normal_flat,
        X_fit_normal_flat,
        validation_split=0.1,
        epochs=AE_EPOCHS,
        batch_size=AE_BATCH,
        callbacks=[ae_cb],
        verbose=2,
    )

    m_mu, m_std, l_mu, l_std = compute_train_normal_stats(encoder, ae, X_fit_normal_flat, AE_BATCH)
    print("Train-normal stats: MSE mu={:.6f} std={:.6f}  ||z|| mu={:.6f} std={:.6f}".format(m_mu, m_std, l_mu, l_std))

    gen, disc, gan_m = build_gan(flat_dim, GAN_NOISE_DIM)
    train_gan(gen, disc, gan_m, X_fit_normal_flat, GAN_EPOCHS, GAN_STEPS_PER_EPOCH, GAN_BATCH)

    best_T, best_CAP, val_fb, val_f1_bin = tune_stage2_threshold_fbeta_gate(
        stage1,
        encoder,
        ae,
        X_val,
        X_val_flat,
        y_val_enc,
        gen,
        normal_idx5,
        m_mu,
        m_std,
        l_mu,
        l_std,
    )

    noise_test = np.random.normal(0, 1, (NUM_GAN_TEST_WINDOWS, GAN_NOISE_DIM)).astype(np.float32)
    X_gan_flat = gen.predict(noise_test, verbose=0).astype(np.float32)
    X_gan_seq = X_gan_flat.reshape(-1, SEQ_LEN, n_feat)

    X_eval_seq = np.concatenate([X_test, X_gan_seq], axis=0)
    X_eval_flat = np.concatenate([X_test_flat, X_gan_flat], axis=0)
    y_eval_true_str = np.concatenate(
        [le5.inverse_transform(y_test_enc), np.full(NUM_GAN_TEST_WINDOWS, "zero_day", dtype=object)]
    )

    p_eval = stage1.predict(X_eval_seq, batch_size=BATCH_SIZE, verbose=0)
    s1_eval = np.argmax(p_eval, axis=1)
    pnorm_eval = p_eval[:, normal_idx5].astype(np.float64)
    sc_eval = combined_anomaly_score(encoder, ae, X_eval_flat, m_mu, m_std, l_mu, l_std, AE_BATCH)

    def stage2_call_zero_day(sc, p_n):
        return (sc > best_T) and (p_n <= best_CAP)

    final_pred = []
    for i in range(len(s1_eval)):
        if s1_eval[i] != normal_idx5:
            final_pred.append(le5.inverse_transform(np.array([s1_eval[i]]))[0])
        else:
            final_pred.append("zero_day" if stage2_call_zero_day(sc_eval[i], pnorm_eval[i]) else "Normal")

    mask_s1_normal = s1_eval == normal_idx5
    y_bin_true, y_bin_pred = [], []
    for i in range(len(y_eval_true_str)):
        if not mask_s1_normal[i]:
            continue
        t = y_eval_true_str[i]
        if t not in ("Normal", "zero_day"):
            continue
        y_bin_true.append(t)
        y_bin_pred.append("zero_day" if stage2_call_zero_day(sc_eval[i], pnorm_eval[i]) else "Normal")
    labels_bin = ["Normal", "zero_day"]
    if len(y_bin_true) > 0:
        print_metrics_block(
            "Stage 2 (imp_2: F{:.1f} + min-recall tune + p(Normal) gate)".format(FBETA_TUNING),
            y_bin_true,
            y_bin_pred,
            labels_bin,
        )
        plot_confusion_heatmap(
            y_bin_true,
            y_bin_pred,
            labels_bin,
            "Stage 2 — F{:.1f} threshold + softmax gate (p(Normal)<=CAP)".format(FBETA_TUNING),
            "midsecond2_cm_stage2.png",
            figsize=(7, 6),
        )

    print_metrics_block("Final hybrid (6 classes)", y_eval_true_str, final_pred, FINAL_LABELS)
    plot_confusion_heatmap(
        y_eval_true_str,
        final_pred,
        FINAL_LABELS,
        "Final hybrid (Stage 2 imp_2: F-beta + gate)",
        "midsecond2_cm_final_hybrid.png",
        figsize=(11, 9),
    )

    print(
        "\n--- midsem_second_stage_imp_2.py complete. T={:.6f} CAP={:.4f} (val F{:.1f}={:.4f}, val F1_bin={:.4f}) ---".format(
            best_T, best_CAP, FBETA_TUNING, val_fb, val_f1_bin
        )
    )
