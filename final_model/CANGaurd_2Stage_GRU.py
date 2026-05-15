# ===============================
# CANGaurd_2Stage_GRU.py
# Same hybrid IDS pipeline as CANGuard_2Stage_LSTM.py, but every recurrent
# block (Stage-1 backbone, Stage-2 sequence autoencoder encoder + decoder)
# has been swapped from LSTM to Bidirectional GRU followed by a
# Multi-Head Self-Attention block. The data loading, feature engineering,
# scoring fusion (mean MSE + max-step MSE + latent Mahalanobis), GAN-based
# OOD probe synthesis, precision-priority F0.5 threshold tuning, and final
# hybrid evaluation are intentionally left UNCHANGED.
#
# Architectural change vs CANGuard_2Stage_LSTM.py:
#   1) Stage 1 backbone:  Conv1D x2 -> BiGRU(96, seq) -> MHA block ->
#                          BiGRU(64, vec) -> Dense head.
#   2) Stage 2 AE encoder: Conv1D x2 -> BiGRU(64, seq) -> MHA block ->
#                          Global average pool -> Dense(latent).
#   3) Stage 2 AE decoder: Dense -> Reshape -> BiGRU(64, seq) -> MHA block ->
#                          Conv1DTranspose x2 -> TimeDistributed(F).
#
# Outputs:
#   cangaurd_gru_cm_stage1.png
#   cangaurd_gru_cm_stage2.png
#   cangaurd_gru_cm_final_hybrid.png
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
    GRU,
    Add,
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Conv1DTranspose,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    LayerNormalization,
    MaxPooling1D,
    MultiHeadAttention,
    Reshape,
    TimeDistributed,
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

# Stage-2 AE hyper-parameters (sequence-aware encoder-decoder)
AE_LATENT_DIM = 32
AE_EPOCHS = 60
AE_PATIENCE = 10
AE_BATCH = 256
AE_ENSEMBLE_SIZE = 3  # number of AEs in the ensemble

# Attention hyper-parameters (shared by Stage 1 and AE encoder/decoder)
ATTN_NUM_HEADS = 4
ATTN_KEY_DIM = 32
ATTN_DROPOUT = 0.1

# Mahalanobis numerical stability
MAHA_EIG_FLOOR = 1e-4
MAHA_SHRINKAGE = 0.05  # convex blend with diagonal of latent variance

# GAN
GAN_NOISE_DIM = 48
GAN_EPOCHS = 30
GAN_STEPS_PER_EPOCH = 130
GAN_BATCH = 128

# Threshold tuning
NUM_GAN_VAL_PROBES = 2500
NUM_NOISE_VAL_PROBES = 1500       # uniform-noise synthetic anomalies
NUM_SHUFFLE_VAL_PROBES = 1500     # feature-shuffled normal anomalies
NUM_GAN_TEST_WINDOWS = 1500       # held-out zero_day windows for test eval

THRESHOLD_GRID_POINTS = 600
F_BETA = 0.5                      # precision-priority
PRECISION_FLOOR = 0.85            # try to keep zero_day precision above this

print("Device / GPU:", tf.config.list_physical_devices("GPU"))

ATTACK_FILE_SPECS = [
    ("DoS_dataset.csv", "dos_attack.csv", "DoS"),
    ("Fuzzy_dataset.csv", "fuzzy_attack.csv", "Fuzzy"),
    ("gear_dataset.csv", "gear_spoofing.csv", "Gear"),
    ("RPM_dataset.csv", "rpm_spoofing.csv", "RPM"),
]

FINAL_LABELS = ["Normal", "DoS", "Fuzzy", "Gear", "RPM", "zero_day"]


# ----------------------------- I/O & utilities -----------------------------

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
        cmap="Purples",
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


# ----------------------------- Data loading -----------------------------

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


# ----------------------------- Models -----------------------------

def attention_block(x, num_heads=ATTN_NUM_HEADS, key_dim=ATTN_KEY_DIM, dropout=ATTN_DROPOUT, name=None):
    """Multi-head self-attention with residual + LayerNorm.

    Input  : (B, T, D) sequence tensor
    Output : (B, T, D) sequence tensor of the same shape
    """
    attn_name = None if name is None else "{}_mha".format(name)
    ln_name = None if name is None else "{}_ln".format(name)
    add_name = None if name is None else "{}_add".format(name)
    attn = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout,
        name=attn_name,
    )(x, x)
    x = Add(name=add_name)([x, attn])
    x = LayerNormalization(name=ln_name)(x)
    return x


def build_stage1_model(seq_len, n_features, num_classes=5):
    """Stage 1 backbone (BiGRU + attention variant of CANGuard_2Stage_LSTM)."""
    inp = Input(shape=(seq_len, n_features))
    x = Conv1D(64, 3, padding="same", activation="relu")(inp)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.25)(x)
    x = Conv1D(128, 3, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.25)(x)
    x = Bidirectional(GRU(96, return_sequences=True))(x)
    x = attention_block(x, name="stage1_attn")
    x = Dropout(0.3)(x)
    x = Bidirectional(GRU(64, return_sequences=False))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.35)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.25)(x)
    out = Dense(num_classes, activation="softmax")(x)
    model = Model(inp, out)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_seq_autoencoder(seq_len, n_features, latent_dim=AE_LATENT_DIM, seed=0):
    """Sequence-aware Conv1D + BiGRU(+attention) bottleneck autoencoder.

    Encoder:  (T, F) -> Conv1D x2 -> BiGRU(seq) -> MHA -> GAP -> Dense(latent)
    Decoder:  Dense -> Reshape(T) -> BiGRU(seq) -> MHA ->
              Conv1DTranspose x2 -> TimeDistributed(F)

    Returns (autoencoder, encoder). Encoder maps a sequence window to a single
    `latent_dim` vector `z` that is used for Mahalanobis scoring.
    """
    tf.random.set_seed(RANDOM_STATE + seed)
    inp = Input(shape=(seq_len, n_features), name="ae_input_seq_{}".format(seed))

    # Encoder
    x = Conv1D(64, 3, padding="same", activation="relu")(inp)
    x = BatchNormalization()(x)
    x = Conv1D(96, 3, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Bidirectional(GRU(64, return_sequences=True))(x)
    x = attention_block(x, name="enc_attn_{}".format(seed))
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.15)(x)
    z = Dense(latent_dim, activation="tanh", name="latent_{}".format(seed))(x)

    # Decoder
    d = Dense(seq_len * 32, activation="relu")(z)
    d = Reshape((seq_len, 32))(d)
    d = Bidirectional(GRU(64, return_sequences=True))(d)
    d = attention_block(d, name="dec_attn_{}".format(seed))
    d = BatchNormalization()(d)
    d = Conv1DTranspose(96, 3, padding="same", activation="relu")(d)
    d = Conv1DTranspose(64, 3, padding="same", activation="relu")(d)
    out = TimeDistributed(Dense(n_features, activation="sigmoid"))(d)

    ae = Model(inp, out, name="seq_ae_{}".format(seed))
    encoder = Model(inp, z, name="seq_encoder_{}".format(seed))
    ae.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    return ae, encoder


def build_gan(flat_dim, noise_dim=GAN_NOISE_DIM):
    """GAN in flat window space, used purely to synthesize OOD probes for tuning."""
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


# ----------------------------- Anomaly scoring -----------------------------

def _predict_in_batches(model, X, batch_size):
    """Predict in batches to avoid GPU OOM on big arrays."""
    n = len(X)
    out = None
    for i in range(0, n, batch_size):
        chunk = X[i : i + batch_size].astype(np.float32)
        y = model.predict(chunk, verbose=0)
        if out is None:
            out = np.empty((n,) + y.shape[1:], dtype=y.dtype)
        out[i : i + len(chunk)] = y
    return out


def _per_window_recon_stats(ae, X_seq, batch_size):
    """Returns (mean_mse, max_step_mse) per window."""
    recon = _predict_in_batches(ae, X_seq, batch_size)
    diff = X_seq.astype(np.float32) - recon  # (N, T, F)
    sq = diff * diff
    step_mse = sq.mean(axis=2)  # (N, T)
    mean_mse = step_mse.mean(axis=1)  # (N,)
    max_mse = step_mse.max(axis=1)    # (N,)
    return mean_mse.astype(np.float64), max_mse.astype(np.float64)


def _fit_latent_mahalanobis(Z_train_normal):
    """Fit mean + shrunk covariance for Mahalanobis distance.

    Uses Ledoit-Wolf style convex blend with the diagonal of the empirical
    variance to keep the precision matrix well-conditioned even for small
    train-normal counts. Returns (mean, precision_matrix).
    """
    mu = Z_train_normal.mean(axis=0)
    Zc = Z_train_normal - mu
    n = max(len(Z_train_normal) - 1, 1)
    cov = (Zc.T @ Zc) / n
    diag = np.diag(np.diag(cov))
    cov_shrunk = (1.0 - MAHA_SHRINKAGE) * cov + MAHA_SHRINKAGE * diag
    # Eigen-floor for numerical stability before inversion.
    w, V = np.linalg.eigh(cov_shrunk)
    w = np.clip(w, MAHA_EIG_FLOOR, None)
    cov_psd = (V * w) @ V.T
    precision = np.linalg.inv(cov_psd)
    return mu.astype(np.float64), precision.astype(np.float64)


def _mahalanobis(Z, mu, precision):
    Zc = Z.astype(np.float64) - mu
    # row-wise quadratic form: sum_i (Zc_i @ P) * Zc_i
    left = Zc @ precision
    d2 = np.einsum("ij,ij->i", left, Zc)
    d2 = np.clip(d2, 0.0, None)
    return np.sqrt(d2)


class StageTwoScorer:
    """Holds the AE ensemble + latent Mahalanobis params + robust-standardization
    parameters needed to compute a single fused anomaly score per window.
    """

    def __init__(self):
        self.aes = []           # list of (ae, encoder)
        self.maha = []          # list of (mu, precision) per AE
        self.norm_params = None  # robust standardization params for fusion

    def fit(self, X_fit_normal_seq, seq_len, n_features):
        Z_per_ae = []
        for k in range(AE_ENSEMBLE_SIZE):
            print("\n-- Training Stage-2 AE #{} (seed={}) --".format(k + 1, RANDOM_STATE + k))
            ae, enc = build_seq_autoencoder(seq_len, n_features, AE_LATENT_DIM, seed=k)
            ae_cb = EarlyStopping(monitor="val_loss", patience=AE_PATIENCE, restore_best_weights=True)
            ae.fit(
                X_fit_normal_seq,
                X_fit_normal_seq,
                validation_split=0.1,
                epochs=AE_EPOCHS,
                batch_size=AE_BATCH,
                callbacks=[ae_cb],
                verbose=2,
            )
            self.aes.append((ae, enc))
            Z = _predict_in_batches(enc, X_fit_normal_seq, AE_BATCH).astype(np.float64)
            mu, prec = _fit_latent_mahalanobis(Z)
            self.maha.append((mu, prec))
            Z_per_ae.append(Z)

        # Calibrate robust standardization on train-normal: median + MAD per
        # raw score, then per-AE scores are averaged at the end.
        s_mean_mse, s_max_mse, s_maha = self._raw_scores(X_fit_normal_seq)

        def robust_params(vec):
            med = float(np.median(vec))
            mad = float(np.median(np.abs(vec - med))) * 1.4826
            if mad < 1e-12:
                mad = float(vec.std() + 1e-8)
            return med, mad

        self.norm_params = {
            "mean_mse": robust_params(s_mean_mse),
            "max_mse": robust_params(s_max_mse),
            "maha": robust_params(s_maha),
        }
        print("\nRobust standardization (median, MAD-scale):")
        for k, v in self.norm_params.items():
            print("  {:8s}: median={:.6f}  scale={:.6f}".format(k, v[0], v[1]))

    def _raw_scores(self, X_seq):
        """Return three per-window raw score vectors averaged over the AE ensemble.

        Each score is computed with each (ae, encoder) and then averaged.
        """
        n = len(X_seq)
        agg_mean = np.zeros(n, dtype=np.float64)
        agg_max = np.zeros(n, dtype=np.float64)
        agg_maha = np.zeros(n, dtype=np.float64)
        for (ae, enc), (mu, prec) in zip(self.aes, self.maha):
            mean_mse, max_mse = _per_window_recon_stats(ae, X_seq, AE_BATCH)
            Z = _predict_in_batches(enc, X_seq, AE_BATCH).astype(np.float64)
            d_maha = _mahalanobis(Z, mu, prec)
            agg_mean += mean_mse
            agg_max += max_mse
            agg_maha += d_maha
        denom = float(max(len(self.aes), 1))
        return agg_mean / denom, agg_max / denom, agg_maha / denom

    def score(self, X_seq):
        """Fused anomaly score per window: average of robust-standardized
        |mean_mse|, |max_step_mse| and Mahalanobis distance in latent.
        """
        s_mean, s_max, s_maha = self._raw_scores(X_seq)
        med_m, sc_m = self.norm_params["mean_mse"]
        med_x, sc_x = self.norm_params["max_mse"]
        med_h, sc_h = self.norm_params["maha"]
        z_m = np.abs(s_mean - med_m) / sc_m
        z_x = np.abs(s_max - med_x) / sc_x
        z_h = np.abs(s_maha - med_h) / sc_h
        return ((z_m + z_x + z_h) / 3.0).astype(np.float64)


# ----------------------- Threshold tuning utilities -----------------------

def _stage1_predicts_normal(stage1, X_seq, normal_idx5):
    preds = np.argmax(stage1.predict(X_seq, batch_size=BATCH_SIZE, verbose=0), axis=1)
    return preds == normal_idx5


def _make_noise_probes(n, seq_len, n_feat):
    """Uniform [0, 1] noise probes - structurally far from CAN traffic."""
    return np.random.uniform(0.0, 1.0, size=(n, seq_len, n_feat)).astype(np.float32)


def _make_shuffle_probes(X_normal_seq, n):
    """Feature-shuffled normal windows: take normal windows and permute feature
    axis independently per time step to destroy the joint feature
    distribution while keeping marginals roughly normal-like. This produces
    a different OOD regime than the GAN.
    """
    if len(X_normal_seq) == 0:
        return np.zeros((0, X_normal_seq.shape[1], X_normal_seq.shape[2]), dtype=np.float32)
    idx = np.random.randint(0, len(X_normal_seq), size=n)
    base = X_normal_seq[idx].copy()
    T = base.shape[1]
    F = base.shape[2]
    for k in range(len(base)):
        for t in range(T):
            perm = np.random.permutation(F)
            base[k, t] = base[k, t, perm]
    return base.astype(np.float32)


def _tune_threshold_precision_priority(scores_neg, scores_pos):
    """Grid-search threshold T maximizing F_beta=0.5 with a soft precision
    floor PRECISION_FLOOR. If no T meets the floor, falls back to plain F0.5.
    Returns (T, fbeta, precision, recall).
    """
    if len(scores_neg) == 0 or len(scores_pos) == 0:
        return 0.5, 0.0, 0.0, 0.0
    y_bin = np.concatenate([np.zeros(len(scores_neg), dtype=np.int32),
                            np.ones(len(scores_pos), dtype=np.int32)])
    s_all = np.concatenate([scores_neg, scores_pos])
    lo = float(np.percentile(s_all, 1.0))
    hi = float(np.percentile(s_all, 99.5))
    if hi <= lo:
        return float(s_all.mean()), 0.0, 0.0, 0.0
    grid = np.linspace(lo, hi, THRESHOLD_GRID_POINTS)

    best_with_floor = None
    best_plain = None
    for t in grid:
        pred = (s_all > t).astype(np.int32)
        fb = fbeta_score(y_bin, pred, beta=F_BETA, zero_division=0)
        prec = precision_score(y_bin, pred, zero_division=0)
        rec = recall_score(y_bin, pred, zero_division=0)
        cand = (fb, prec, rec, float(t))
        if best_plain is None or fb > best_plain[0]:
            best_plain = cand
        if prec >= PRECISION_FLOOR:
            if best_with_floor is None or fb > best_with_floor[0]:
                best_with_floor = cand
    chosen = best_with_floor if best_with_floor is not None else best_plain
    fb, prec, rec, t = chosen
    print(
        "Threshold tuned (precision-priority, F0.5):"
        "  T={:.6f}  F0.5={:.4f}  precision={:.4f}  recall={:.4f}  used_floor={}".format(
            t, fb, prec, rec, best_with_floor is not None
        )
    )
    return t, fb, prec, rec


# ----------------------------- Main pipeline -----------------------------

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

    # MinMaxScaler is fit on full data (same as midsem variant - no leakage of
    # labels, only feature ranges)
    X_flat = X_w.reshape(len(X_w), flat_dim)
    scaler = MinMaxScaler()
    X_flat_s = scaler.fit_transform(X_flat).astype(np.float32)
    X_s = X_flat_s.reshape(len(X_w), SEQ_LEN, n_feat)
    y_cat5 = to_categorical(y_enc, num_classes=5)

    # Stratified outer split + inner val split (same as midsem variant).
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
    X_fit_normal_seq = X_fit[is_fit_normal]
    X_fit_normal_flat = X_fit_flat[is_fit_normal]
    print("AE-fit normal windows:", len(X_fit_normal_seq))

    classes = np.unique(y_fit_enc)
    cw = compute_class_weight("balanced", classes=classes, y=y_fit_enc)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, cw)}

    # ----- Stage 1 -----
    print("\n========== Training Stage 1 ==========")
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
    plot_confusion_heatmap(
        y_s1_true,
        y_s1_pred,
        labels5,
        "Stage 1 (CANGaurd_2Stage_GRU)",
        "cangaurd_gru_cm_stage1.png",
    )

    # ----- Stage 2: AE ensemble fit on train-normal windows -----
    print("\n========== Training Stage 2 (AE ensemble + Mahalanobis fusion) ==========")
    scorer = StageTwoScorer()
    scorer.fit(X_fit_normal_seq, SEQ_LEN, n_feat)

    # ----- GAN (used only to synthesize OOD probes for threshold tuning) -----
    print("\n========== Training GAN (synthetic OOD probes for tuning) ==========")
    gen, disc, gan_m = build_gan(flat_dim, GAN_NOISE_DIM)
    train_gan(gen, disc, gan_m, X_fit_normal_flat, GAN_EPOCHS, GAN_STEPS_PER_EPOCH, GAN_BATCH)

    # ----- Build diverse synthetic anomaly set on validation -----
    print("\n========== Tuning Stage-2 threshold (precision-priority F0.5) ==========")

    # Negatives: validation Normal windows where Stage 1 also predicts Normal.
    s1_val = np.argmax(stage1.predict(X_val, batch_size=BATCH_SIZE, verbose=0), axis=1)
    mask_val_normal = (y_val_enc == normal_idx5) & (s1_val == normal_idx5)
    X_val_neg = X_val[mask_val_normal]
    if len(X_val_neg) < 200:
        X_val_neg = X_val[s1_val == normal_idx5]
    scores_neg = scorer.score(X_val_neg) if len(X_val_neg) > 0 else np.array([])
    print("Validation negatives (Normal, Stage1=Normal):", len(X_val_neg))

    # Positives mix: GAN + uniform noise + feature-shuffled normal, all
    # restricted to those that Stage 1 predicts as Normal (otherwise they
    # don't even reach Stage 2 at inference time).
    noise_gan = np.random.normal(0, 1, (NUM_GAN_VAL_PROBES, GAN_NOISE_DIM)).astype(np.float32)
    X_gan_flat_val = gen.predict(noise_gan, verbose=0).astype(np.float32)
    X_gan_seq_val = X_gan_flat_val.reshape(-1, SEQ_LEN, n_feat)
    X_noise_seq_val = _make_noise_probes(NUM_NOISE_VAL_PROBES, SEQ_LEN, n_feat)
    X_shuf_seq_val = _make_shuffle_probes(X_fit_normal_seq, NUM_SHUFFLE_VAL_PROBES)
    X_pos_all = np.concatenate([X_gan_seq_val, X_noise_seq_val, X_shuf_seq_val], axis=0)

    mask_pos_s1_normal = _stage1_predicts_normal(stage1, X_pos_all, normal_idx5)
    X_val_pos = X_pos_all[mask_pos_s1_normal]
    if len(X_val_pos) < 200:
        X_val_pos = X_pos_all
    scores_pos = scorer.score(X_val_pos) if len(X_val_pos) > 0 else np.array([])
    print(
        "Validation positives (synthetic OOD, Stage1=Normal):"
        " gan={}  noise={}  shuffle={}  total_kept={}".format(
            NUM_GAN_VAL_PROBES, NUM_NOISE_VAL_PROBES, NUM_SHUFFLE_VAL_PROBES, len(X_val_pos)
        )
    )

    best_T, val_fb, val_p, val_r = _tune_threshold_precision_priority(scores_neg, scores_pos)

    # ----- Test-time evaluation -----
    print("\n========== Evaluating on held-out test set ==========")
    noise_test = np.random.normal(0, 1, (NUM_GAN_TEST_WINDOWS, GAN_NOISE_DIM)).astype(np.float32)
    X_gan_flat_test = gen.predict(noise_test, verbose=0).astype(np.float32)
    X_gan_seq_test = X_gan_flat_test.reshape(-1, SEQ_LEN, n_feat)

    X_eval_seq = np.concatenate([X_test, X_gan_seq_test], axis=0)
    y_eval_true_str = np.concatenate(
        [le5.inverse_transform(y_test_enc), np.full(NUM_GAN_TEST_WINDOWS, "zero_day", dtype=object)]
    )

    s1_eval = np.argmax(stage1.predict(X_eval_seq, batch_size=BATCH_SIZE, verbose=0), axis=1)
    sc_eval = scorer.score(X_eval_seq)

    final_pred = []
    for i in range(len(s1_eval)):
        if s1_eval[i] != normal_idx5:
            final_pred.append(le5.inverse_transform(np.array([s1_eval[i]]))[0])
        else:
            final_pred.append("zero_day" if sc_eval[i] > best_T else "Normal")

    # Stage-2 binary view (Normal vs zero_day) on Stage1=Normal slice
    mask_s1_normal = s1_eval == normal_idx5
    y_bin_true, y_bin_pred = [], []
    for i in range(len(y_eval_true_str)):
        if not mask_s1_normal[i]:
            continue
        t = y_eval_true_str[i]
        if t not in ("Normal", "zero_day"):
            continue
        y_bin_true.append(t)
        y_bin_pred.append("zero_day" if sc_eval[i] > best_T else "Normal")
    labels_bin = ["Normal", "zero_day"]
    if len(y_bin_true) > 0:
        print_metrics_block(
            "Stage 2 (CANGaurd_2Stage_GRU: seq-AE ensemble + Mahalanobis, F0.5 threshold)",
            y_bin_true,
            y_bin_pred,
            labels_bin,
        )
        plot_confusion_heatmap(
            y_bin_true,
            y_bin_pred,
            labels_bin,
            "Stage 2 — CANGaurd_2Stage_GRU (seq-AE ensemble + Mahalanobis)",
            "cangaurd_gru_cm_stage2.png",
            figsize=(7, 6),
        )

    print_metrics_block("Final hybrid (6 classes) - CANGaurd_2Stage_GRU", y_eval_true_str, final_pred, FINAL_LABELS)
    plot_confusion_heatmap(
        y_eval_true_str,
        final_pred,
        FINAL_LABELS,
        "Final hybrid — CANGaurd_2Stage_GRU",
        "cangaurd_gru_cm_final_hybrid.png",
        figsize=(11, 9),
    )

    print(
        "\n--- CANGaurd_2Stage_GRU.py complete."
        "  Best Stage-2 T={:.6f}  val F0.5={:.4f}  val precision={:.4f}  val recall={:.4f} ---".format(
            best_T, val_fb, val_p, val_r
        )
    )
