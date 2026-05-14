# ===============================
# midsem_second_stage_aae.py
# Same hybrid IDS as midsem_second_stage_imp.py, but Stage 2 is an Adversarial Autoencoder (AAE):
#   - Encoder: flat window -> latent z
#   - Decoder: z -> reconstruction (minimize MSE on train-Normal)
#   - Latent Discriminator: classifies z ~ N(0,I) (real) vs encoder(x) (fake); encoder+decoder
#     are trained to fool D while keeping reconstruction low (adversarial + recon).
#   - Anomaly score on inference: max( z-score(MSE), z-score(D(E(x))) ) vs train-Normal stats
#     (same spirit as imp's max(MSE, ||z||) but second term uses D on encoded z).
#   - Flat-space GAN probes + val F1 threshold on combined score (unchanged from imp).
#
# Outputs: midsecond_aae_cm_stage1.png, midsecond_aae_cm_stage2.png, midsecond_aae_cm_final_hybrid.png
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

AAE_LATENT_DIM = 40
AAE_PRETRAIN_EPOCHS = 12
AAE_PRETRAIN_PATIENCE = 5
AAE_ADV_EPOCHS = 35
AAE_STEPS_PER_EPOCH = 130
AAE_BATCH = 256
AAE_ADV_WEIGHT = 0.05

GAN_NOISE_DIM = 48
GAN_EPOCHS = 28
GAN_STEPS_PER_EPOCH = 130
GAN_BATCH = 128
NUM_GAN_VAL_PROBES = 2500
NUM_GAN_TEST_WINDOWS = 1500

THRESHOLD_GRID_POINTS = 400

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


def build_aae_encoder_decoder(flat_dim, latent_dim=AAE_LATENT_DIM):
    """Encoder x->z (linear z), Decoder z->x_hat (sigmoid)."""
    enc_in = Input(shape=(flat_dim,), name="aae_flat_in")
    x = Dense(256, activation="relu")(enc_in)
    x = BatchNormalization()(x)
    x = Dropout(0.12)(x)
    x = Dense(128, activation="relu")(x)
    z = Dense(latent_dim, activation="linear", name="aae_z")(x)

    dec_in = Input(shape=(latent_dim,), name="aae_z_in")
    y = Dense(128, activation="relu")(dec_in)
    y = BatchNormalization()(y)
    y = Dropout(0.12)(y)
    y = Dense(256, activation="relu")(y)
    x_hat = Dense(flat_dim, activation="sigmoid", name="aae_recon")(y)

    encoder = Model(enc_in, z, name="aae_encoder")
    decoder = Model(dec_in, x_hat, name="aae_decoder")
    autoencoder = Model(enc_in, decoder(encoder(enc_in)), name="aae_autoencoder")
    autoencoder.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    return encoder, decoder, autoencoder


def build_latent_discriminator(latent_dim=AAE_LATENT_DIM):
    """Binary: 1 = sample from N(0,I), 0 = encoder(x)."""
    d_in = Input(shape=(latent_dim,))
    d = Dense(128, activation="relu")(d_in)
    d = Dropout(0.3)(d)
    d = Dense(64, activation="relu")(d)
    d = Dropout(0.2)(d)
    d = Dense(1, activation="sigmoid")(d)
    disc = Model(d_in, d, name="aae_latent_discriminator")
    return disc


def pretrain_aae_autoencoder(autoencoder, X_normal_flat, epochs, batch_size, patience):
    if len(X_normal_flat) < batch_size:
        return
    cb = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    autoencoder.fit(
        X_normal_flat,
        X_normal_flat,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[cb],
        verbose=2,
    )


def train_aae_adversarial(
    encoder,
    decoder,
    disc_z,
    X_normal_flat,
    epochs,
    steps_per_epoch,
    batch_size,
    adv_weight,
    latent_dim,
):
    """Alternate D on (N(0,I) vs E(x)) and (E+D) with recon + BCE(D(E(x)),1)."""
    n = len(X_normal_flat)
    if n < batch_size:
        return
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    opt_d = Adam(learning_rate=2e-4, beta_1=0.5)
    opt_eg = Adam(learning_rate=1e-3)

    for _ep in range(epochs):
        for _ in range(steps_per_epoch):
            idx = np.random.randint(0, n, size=batch_size)
            xb = tf.constant(X_normal_flat[idx].astype(np.float32))

            with tf.GradientTape() as tape_d:
                z_e = encoder(xb, training=True)
                z_r = tf.random.normal((batch_size, latent_dim), dtype=tf.float32)
                dr = disc_z(z_r, training=True)
                df = disc_z(tf.stop_gradient(z_e), training=True)
                loss_d = bce(tf.ones((batch_size, 1)), dr) + bce(tf.zeros((batch_size, 1)), df)
            grads_d = tape_d.gradient(loss_d, disc_z.trainable_variables)
            opt_d.apply_gradients(zip(grads_d, disc_z.trainable_variables))

            disc_z.trainable = False
            with tf.GradientTape() as tape_e:
                z2 = encoder(xb, training=True)
                xh = decoder(z2, training=True)
                loss_r = tf.reduce_mean(tf.reduce_mean(tf.square(xb - xh), axis=-1))
                d_on_z = disc_z(z2, training=True)
                loss_a = bce(tf.ones((batch_size, 1)), d_on_z)
                loss_eg = loss_r + adv_weight * loss_a
            vars_eg = encoder.trainable_variables + decoder.trainable_variables
            grads_e = tape_e.gradient(loss_eg, vars_eg)
            opt_eg.apply_gradients(zip(grads_e, vars_eg))
            disc_z.trainable = True


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


def compute_train_normal_stats_aae(encoder, decoder, disc_z, X_normal_flat, batch_size):
    """Mean/std of MSE recon and D(E(x)) on train-normal for z-scoring."""
    n = len(X_normal_flat)
    mses, douts = [], []
    for i in range(0, n, batch_size):
        chunk = X_normal_flat[i : i + batch_size].astype(np.float32)
        z = encoder.predict(chunk, verbose=0)
        recon = decoder.predict(z, verbose=0)
        dscore = disc_z.predict(z, verbose=0).reshape(-1)
        mses.append(np.mean(np.square(chunk - recon), axis=1))
        douts.append(dscore)
    mse_all = np.concatenate(mses)
    d_all = np.concatenate(douts)
    m_mu, m_std = float(mse_all.mean()), float(mse_all.std() + 1e-8)
    d_mu, d_std = float(d_all.mean()), float(d_all.std() + 1e-8)
    return m_mu, m_std, d_mu, d_std


def combined_anomaly_score_aae(encoder, decoder, disc_z, X_flat, m_mu, m_std, d_mu, d_std, batch_size):
    """max(|z_MSE|, |z_D|) — OOD often shifts recon or latent-adversarial score."""
    n = len(X_flat)
    scores = []
    for i in range(0, n, batch_size):
        chunk = X_flat[i : i + batch_size].astype(np.float32)
        z = encoder.predict(chunk, verbose=0)
        recon = decoder.predict(z, verbose=0)
        dscore = disc_z.predict(z, verbose=0).reshape(-1)
        mse = np.mean(np.square(chunk - recon), axis=1)
        zs_m = np.abs(mse - m_mu) / m_std
        zs_d = np.abs(dscore - d_mu) / d_std
        scores.append(np.maximum(zs_m, zs_d))
    return np.concatenate(scores)


def tune_stage2_threshold_f1_aae(
    stage1,
    encoder,
    decoder,
    disc_z,
    X_val_seq,
    X_val_flat,
    y_val_enc,
    gen,
    normal_idx5,
    m_mu,
    m_std,
    d_mu,
    d_std,
):
    s1 = np.argmax(stage1.predict(X_val_seq, batch_size=BATCH_SIZE, verbose=0), axis=1)
    sc_norm = combined_anomaly_score_aae(encoder, decoder, disc_z, X_val_flat, m_mu, m_std, d_mu, d_std, AAE_BATCH)

    mask_n = (y_val_enc == normal_idx5) & (s1 == normal_idx5)
    scores_neg = sc_norm[mask_n]
    if len(scores_neg) < 50:
        scores_neg = sc_norm[s1 == normal_idx5]

    noise_v = np.random.normal(0, 1, (NUM_GAN_VAL_PROBES, GAN_NOISE_DIM)).astype(np.float32)
    X_gan_v = gen.predict(noise_v, verbose=0).astype(np.float32)
    X_gan_seq = X_gan_v.reshape(-1, SEQ_LEN, X_val_seq.shape[2])
    s1_g = np.argmax(stage1.predict(X_gan_seq, batch_size=BATCH_SIZE, verbose=0), axis=1)
    sc_gan = combined_anomaly_score_aae(encoder, decoder, disc_z, X_gan_v, m_mu, m_std, d_mu, d_std, AAE_BATCH)
    mask_g = s1_g == normal_idx5
    scores_pos = sc_gan[mask_g]
    if len(scores_pos) < 50:
        scores_pos = sc_gan

    y_bin = np.concatenate([np.zeros(len(scores_neg), dtype=np.int32), np.ones(len(scores_pos), dtype=np.int32)])
    scores_all = np.concatenate([scores_neg, scores_pos])

    lo = float(np.min(scores_all))
    hi = float(np.max(scores_all))
    if hi <= lo:
        return 0.5 * (lo + hi), 0.0

    candidates = np.linspace(lo, hi, THRESHOLD_GRID_POINTS)
    best_t, best_f1 = None, -1.0
    for t in candidates:
        pred = (scores_all > t).astype(np.int32)
        f1 = f1_score(y_bin, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    print(
        "Stage-2 threshold (val F1 tuned, AAE score): T={:.6f}  val F1={:.4f}  |neg|={} |pos|={}".format(
            best_t, best_f1, len(scores_neg), len(scores_pos)
        )
    )
    return best_t, best_f1


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
    print("AAE fit normal windows:", len(X_fit_normal_flat))

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
    plot_confusion_heatmap(y_s1_true, y_s1_pred, labels5, "Stage 1 (AAE Stage-2 pipeline)", "midsecond_aae_cm_stage1.png")

    encoder, decoder, autoencoder = build_aae_encoder_decoder(flat_dim, AAE_LATENT_DIM)
    print("AAE pretrain (reconstruction only)...")
    pretrain_aae_autoencoder(
        autoencoder,
        X_fit_normal_flat,
        AAE_PRETRAIN_EPOCHS,
        AAE_BATCH,
        AAE_PRETRAIN_PATIENCE,
    )

    disc_z = build_latent_discriminator(AAE_LATENT_DIM)
    disc_z.compile(optimizer=Adam(learning_rate=2e-4, beta_1=0.5), loss="binary_crossentropy")
    print("AAE adversarial phase (latent D vs N(0,I))...")
    train_aae_adversarial(
        encoder,
        decoder,
        disc_z,
        X_fit_normal_flat,
        AAE_ADV_EPOCHS,
        AAE_STEPS_PER_EPOCH,
        AAE_BATCH,
        AAE_ADV_WEIGHT,
        AAE_LATENT_DIM,
    )

    m_mu, m_std, d_mu, d_std = compute_train_normal_stats_aae(
        encoder, decoder, disc_z, X_fit_normal_flat, AAE_BATCH
    )
    print(
        "Train-normal stats: MSE mu={:.6f} std={:.6f}  D(z) mu={:.6f} std={:.6f}".format(m_mu, m_std, d_mu, d_std)
    )

    gen, disc, gan_m = build_gan(flat_dim, GAN_NOISE_DIM)
    train_gan(gen, disc, gan_m, X_fit_normal_flat, GAN_EPOCHS, GAN_STEPS_PER_EPOCH, GAN_BATCH)

    best_T, val_f1 = tune_stage2_threshold_f1_aae(
        stage1,
        encoder,
        decoder,
        disc_z,
        X_val,
        X_val_flat,
        y_val_enc,
        gen,
        normal_idx5,
        m_mu,
        m_std,
        d_mu,
        d_std,
    )

    noise_test = np.random.normal(0, 1, (NUM_GAN_TEST_WINDOWS, GAN_NOISE_DIM)).astype(np.float32)
    X_gan_flat = gen.predict(noise_test, verbose=0).astype(np.float32)
    X_gan_seq = X_gan_flat.reshape(-1, SEQ_LEN, n_feat)

    X_eval_seq = np.concatenate([X_test, X_gan_seq], axis=0)
    X_eval_flat = np.concatenate([X_test_flat, X_gan_flat], axis=0)
    y_eval_true_str = np.concatenate(
        [le5.inverse_transform(y_test_enc), np.full(NUM_GAN_TEST_WINDOWS, "zero_day", dtype=object)]
    )

    s1_eval = np.argmax(stage1.predict(X_eval_seq, batch_size=BATCH_SIZE, verbose=0), axis=1)
    sc_eval = combined_anomaly_score_aae(
        encoder, decoder, disc_z, X_eval_flat, m_mu, m_std, d_mu, d_std, AAE_BATCH
    )

    final_pred = []
    for i in range(len(s1_eval)):
        if s1_eval[i] != normal_idx5:
            final_pred.append(le5.inverse_transform(np.array([s1_eval[i]]))[0])
        else:
            final_pred.append("zero_day" if sc_eval[i] > best_T else "Normal")

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
        print_metrics_block("Stage 2 (AAE: recon + D(z) score, val F1 threshold)", y_bin_true, y_bin_pred, labels_bin)
        plot_confusion_heatmap(
            y_bin_true,
            y_bin_pred,
            labels_bin,
            "Stage 2 — AAE max(z_MSE, z_D(E(x))), T from val F1",
            "midsecond_aae_cm_stage2.png",
            figsize=(7, 6),
        )

    print_metrics_block("Final hybrid (6 classes)", y_eval_true_str, final_pred, FINAL_LABELS)
    plot_confusion_heatmap(
        y_eval_true_str,
        final_pred,
        FINAL_LABELS,
        "Final hybrid (Stage 2 AAE)",
        "midsecond_aae_cm_final_hybrid.png",
        figsize=(11, 9),
    )

    print("\n--- midsem_second_stage_aae.py complete. Best Stage-2 T={:.6f} (val F1={:.4f}) ---".format(best_T, val_f1))
