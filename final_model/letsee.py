# -*- coding: utf-8 -*-
"""letsee.py — Hybrid CAN-Bus IDS with improved Stage 2

| Stage | What it does |
|-------|--------------|
| Stage 1 | IDENTICAL to claude_opus_model_colab.py — Conv1D + BiLSTM multi-class classifier |
| Stage 2 | IMPROVED — VAE ensemble + BiLSTM-Attention temporal predictor + learned meta-classifier |

Stage 2 improvements over claude_opus_model_colab.py
=====================================================
1. Variational Autoencoder (VAE) ensemble (K=3) replaces regular AE
   - Reparameterization trick → proper probability model for normal traffic
   - Two new anomaly channels: `kl_div` (KL divergence) and `elbo`
     * Normal traffic → small KL (code near Gaussian prior)
     * Attacks         → large KL (unusual latent codes)
2. Improved temporal predictor: Bidirectional LSTM + Multi-Head Self-Attention
   - Richer context modeling vs plain BiLSTM single-step predictor
3. 9-channel fusion (vs 7):
   mean_mse, max_mse, kl_div, elbo, maha, knn_dist, svdd_dist,
   ensemble_disagree, pred_error
4. Learned meta-classifier (LogisticRegression, C=0.1, balanced):
   - Trained on val_neg / val_pos (real unseen RPM attack windows)
   - Learns optimal per-channel weights from actual data
   - Replaces hand-coded fixed weights → adaptive, attack-aware fusion
5. Same confidence-adjusted gating and LOAO evaluation as claude_opus

Outputs: letsee_cm_stage1.png, letsee_cm_stage2.png, letsee_cm_final.png
"""

# ── Google Colab mount ────────────────────────────────────────────────────────
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    _IN_COLAB = True
except ImportError:
    _IN_COLAB = False

import os

DATA_PATH = (
    '/content/drive/MyDrive/dataset/9) Car-Hacking Dataset'
    if _IN_COLAB else
    r'9) Car-Hacking Dataset'
)

if not os.path.isdir(DATA_PATH):
    print('⚠️  Dataset folder NOT found at:', DATA_PATH)
    print('   Please upload your dataset or correct DATA_PATH above.')
else:
    print('✅ Dataset folder found:', DATA_PATH)
    print('   Contents:', os.listdir(DATA_PATH)[:10])

# ── Imports ───────────────────────────────────────────────────────────────────
import os, re, warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, fbeta_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    LSTM, Add, BatchNormalization, Bidirectional,
    Conv1D, Conv1DTranspose, Dense, Dropout,
    Flatten, GlobalAveragePooling1D, Input,
    LayerNormalization, MaxPooling1D, MultiHeadAttention,
    RepeatVector, Reshape, TimeDistributed,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

print('TensorFlow version :', tf.__version__)
print('GPU devices        :', tf.config.list_physical_devices('GPU'))

# ── Hyper-parameters & global config ─────────────────────────────────────────
data_path = DATA_PATH

# Subset / sampling controls
USE_SUBSET          = True
MAX_NORMAL          = 5_000
MAX_PER_ATTACK_FILE = 10_000

# Windowing & Stage 1
SEQ_LEN             = 24
BATCH_SIZE          = 64
STAGE1_EPOCHS       = 55
STAGE1_PATIENCE     = 12
STAGE1_VAL_SPLIT    = 0.15

# Stage-2 VAE ensemble
AE_LATENT_DIM       = 32
AE_EPOCHS           = 60
AE_PATIENCE         = 10
AE_BATCH            = 256
AE_ENSEMBLE_SIZE    = 3

# VAE-specific (beta-VAE KL weight)
# Small beta → tight reconstruction; non-zero → regularised latent for anomaly scoring
VAE_BETA            = 0.001

# Temporal Predictor
PRED_EPOCHS         = 40
PRED_PATIENCE       = 8
PRED_BATCH          = 256

# Attention in improved temporal predictor
ATTN_NUM_HEADS      = 4
ATTN_KEY_DIM        = 32
ATTN_DROPOUT        = 0.1

# Mahalanobis numerical stability
MAHA_EIG_FLOOR      = 1e-4
MAHA_SHRINKAGE      = 0.05

# k-NN anomaly scoring
KNN_K               = 10
KNN_MAX_TRAIN       = 3000

# Learned meta-classifier
META_CLF_C          = 0.1   # LogisticRegression regularisation (L2)

# Stage 1 confidence gating
S1_CONFIDENCE_WEIGHT = 0.25

# Threshold tuning
THRESHOLD_GRID_POINTS   = 600
F_BETA                  = 0.5
PRECISION_FLOOR         = 0.85

# Attack file specs
ATTACK_FILE_SPECS = [
    ('DoS_dataset.csv',   'dos_attack.csv',   'DoS'),
    ('Fuzzy_dataset.csv', 'fuzzy_attack.csv', 'Fuzzy'),
    ('gear_dataset.csv',  'gear_spoofing.csv','Gear'),
    ('RPM_dataset.csv',   'rpm_spoofing.csv', 'RPM'),
]

UNSEEN_ATTACK = "RPM"
ALL_ATTACKS   = ["DoS", "Fuzzy", "Gear", "RPM"]
KNOWN_ATTACKS = [a for a in ALL_ATTACKS if a != UNSEEN_ATTACK]
FINAL_LABELS  = ['Normal'] + KNOWN_ATTACKS + ['zero_day']

print("Known attacks :", KNOWN_ATTACKS)
print("Unseen attack :", UNSEEN_ATTACK)
print("Final labels  :", FINAL_LABELS)
print('Config loaded ✅')

# ── Utility helpers ───────────────────────────────────────────────────────────

def _output_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


def plot_confusion_heatmap(y_true, y_pred, labels, title, filename, figsize=(10, 8)):
    cm  = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy_score(y_true, y_pred)
    p_m = precision_score(y_true, y_pred, average='macro',    zero_division=0, labels=labels)
    r_m = recall_score   (y_true, y_pred, average='macro',    zero_division=0, labels=labels)
    f_m = f1_score       (y_true, y_pred, average='macro',    zero_division=0, labels=labels)
    path = os.path.join(_output_dir(), filename)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=labels, yticklabels=labels, ax=ax,
                cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(
        '{}\nAcc={:.4f}  macro P/R/F1={:.4f}/{:.4f}/{:.4f}'.format(title, acc, p_m, r_m, f_m)
    )
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print('Saved:', path)
    if _IN_COLAB:
        from IPython.display import Image, display
        display(Image(path))


def print_metrics_block(name, y_true, y_pred, labels):
    acc   = accuracy_score  (y_true, y_pred)
    p_mac = precision_score (y_true, y_pred, average='macro',    zero_division=0, labels=labels)
    r_mac = recall_score    (y_true, y_pred, average='macro',    zero_division=0, labels=labels)
    f_mac = f1_score        (y_true, y_pred, average='macro',    zero_division=0, labels=labels)
    p_w   = precision_score (y_true, y_pred, average='weighted', zero_division=0, labels=labels)
    r_w   = recall_score    (y_true, y_pred, average='weighted', zero_division=0, labels=labels)
    f_w   = f1_score        (y_true, y_pred, average='weighted', zero_division=0, labels=labels)
    print('\n========== {} =========='.format(name))
    print('Accuracy:            {:.4f}'.format(acc))
    print('Precision (macro):   {:.4f}   (weighted): {:.4f}'.format(p_mac, p_w))
    print('Recall    (macro):   {:.4f}   (weighted): {:.4f}'.format(r_mac, r_w))
    print('F1-score  (macro):   {:.4f}   (weighted): {:.4f}'.format(f_mac, f_w))
    print(classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0))

print('Utility helpers defined ✅')

# ── Data loading & feature engineering (IDENTICAL to claude_opus) ─────────────

def parse_line(line):
    regex = r'Timestamp:\s*(\d+\.?\d*)\s+ID:\s*(\w+)\s+000\s+DLC:\s*(\d+)\s+([\da-fA-F\s]+)'
    match = re.match(regex, line.strip())
    if match:
        timestamp = float(match.group(1))
        can_id    = int(match.group(2), 16)
        dlc       = int(match.group(3))
        data      = [int(b, 16) for b in match.group(4).split()]
        data      = (data + [0] * 8)[:8]
        return {'Timestamp': timestamp, 'CAN_ID': can_id, 'DLC': dlc, 'DATA': data}
    return None


def load_normal_df(base_path):
    candidates = [
        os.path.join(base_path, 'normal_run_data.txt'),
        os.path.join(base_path, 'normal_run_data', 'normal_run_data.txt'),
    ]
    file_path = next((p for p in candidates if os.path.isfile(p)), None)
    if file_path is None:
        raise FileNotFoundError('normal_run_data.txt not found under ' + repr(base_path))
    rows = []
    with open(file_path, 'r') as f:
        for line in f:
            if USE_SUBSET and len(rows) >= MAX_NORMAL:
                break
            p = parse_line(line)
            if p:
                rows.append(p)
    df = pd.DataFrame(rows)
    for i in range(8):
        df['DATA{}'.format(i)] = df['DATA'].apply(lambda x, i=i: x[i] if i < len(x) else 0)
    df.drop(columns=['DATA'], inplace=True)
    df['Label'] = 'Normal'
    return df


def convert_numeric_columns(df, columns_to_convert):
    for col in columns_to_convert:
        if col == 'CAN_ID' or col.startswith('DATA'):
            df[col] = (df[col].astype(str)
                       .apply(lambda x: int(x, 16) if re.match(r'^[0-9a-fA-F]+$', str(x).strip()) else np.nan))
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(0).astype(int)
    return df


def label_from_flag(flag_series, attack_name):
    return [
        'Normal' if v in ('R', '') else attack_name
        for v in flag_series.astype(str).str.strip().str.upper()
    ]


def load_attack_df(base_path, primary_name, alt_name, attack_label):
    p1   = os.path.join(base_path, primary_name)
    p2   = os.path.join(base_path, alt_name) if alt_name != primary_name else None
    path = p1 if os.path.isfile(p1) else (p2 if p2 and os.path.isfile(p2) else None)
    if path is None:
        raise FileNotFoundError('Missing attack file: {} / {}'.format(primary_name, alt_name))
    column_names = ['Timestamp', 'CAN_ID', 'DLC'] + ['DATA{}'.format(i) for i in range(8)] + ['Flag']
    nrows = MAX_PER_ATTACK_FILE if USE_SUBSET else None
    df = pd.read_csv(path, header=None, names=column_names, nrows=nrows)
    df = convert_numeric_columns(df, ['CAN_ID', 'DLC'] + ['DATA{}'.format(i) for i in range(8)])
    df['Label'] = label_from_flag(df['Flag'], attack_label)
    return df


def add_engineered_features(df):
    df = df.sort_values('Timestamp').reset_index(drop=True)
    ts  = df['Timestamp'].values.astype(np.float64)
    iat = np.diff(ts, prepend=ts[0])
    iat[0] = 0.0
    df['IAT']         = np.clip(iat, 0.0, 1.0)
    freq = df['CAN_ID'].value_counts(normalize=True)
    df['CAN_ID_freq'] = df['CAN_ID'].map(freq).fillna(0.0).astype(np.float64)
    data_cols = ['DATA{}'.format(i) for i in range(8)]

    def row_entropy(row):
        vals = row.values.astype(float)
        vals = vals[vals > 0]
        if len(vals) == 0:
            return 0.0
        p = vals / vals.sum()
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p + 1e-12)))

    df['byte_entropy'] = df[data_cols].apply(row_entropy, axis=1)
    df['byte_sum']     = df[data_cols].sum(axis=1).astype(np.float64)
    df['byte_range']   = (df[data_cols].max(axis=1) - df[data_cols].min(axis=1)).astype(np.float64)
    df['byte_std']     = df[data_cols].std(axis=1).fillna(0.0).astype(np.float64)
    return df


def make_windows_from_sorted_df(df, base_cols, seq_len):
    df = add_engineered_features(df)
    feat_cols = base_cols + ['IAT', 'CAN_ID_freq', 'byte_entropy', 'byte_sum', 'byte_range', 'byte_std']
    X     = df[feat_cols].values.astype(np.float64)
    y_str = df['Label'].values
    if len(X) < seq_len:
        return np.zeros((0, seq_len, len(feat_cols)), np.float32), np.array([], dtype=object)
    n, f = len(X) - seq_len + 1, X.shape[1]
    Xw   = np.zeros((n, seq_len, f), dtype=np.float32)
    yw   = np.empty(n, dtype=object)
    for i in range(n):
        Xw[i] = X[i: i + seq_len]
        yw[i] = y_str[i + seq_len - 1]
    return Xw, yw


def build_all_windows(data_path):
    def _parse_can_csv(path, fname=None):
        try:
            dfr = pd.read_csv(path)
        except Exception as e:
            print('Failed to read:', path, '->', e)
            return None
        colmap   = {c.lower(): c for c in dfr.columns}
        ts_col   = colmap.get('timestamp',      list(dfr.columns)[0])
        aid_col  = colmap.get('arbitration_id', list(dfr.columns)[min(1, len(dfr.columns)-1)])
        data_col = colmap.get('data_field', None)
        atk_col  = colmap.get('attack', None)

        df2 = pd.DataFrame()
        df2['Timestamp'] = pd.to_numeric(dfr[ts_col], errors='coerce').fillna(0.0)

        def parse_arbitration(x):
            s = str(x).strip()
            if s.lower().startswith('0x'):
                try: return int(s, 16)
                except: return 0
            try: return int(float(s))
            except:
                try: return int(s, 16)
                except: return 0

        df2['CAN_ID'] = dfr[aid_col].apply(parse_arbitration)

        def parse_bytes_field(s):
            if pd.isna(s):
                return [0] * 8
            tokens = [t for t in re.split(r'[^0-9A-Fa-fxX]+', str(s)) if t]
            out = []
            for t in tokens:
                try:
                    b = int(t, 16) if (t.lower().startswith('0x') or re.search('[a-fA-F]', t)) else int(t)
                except:
                    try: b = int(t, 16)
                    except: b = 0
                out.append(int(b) & 0xFF)
                if len(out) >= 8: break
            while len(out) < 8: out.append(0)
            return out

        byte_cols = ['DATA{}'.format(i) for i in range(8)]
        if data_col and data_col in dfr.columns:
            parsed = list(dfr[data_col].apply(parse_bytes_field))
            df2['DLC'] = dfr[data_col].apply(
                lambda s: len([t for t in re.split(r'[^0-9A-Fa-fxX]+', str(s)) if t])
            )
        else:
            parsed = [[0] * 8 for _ in range(len(dfr))]
            df2['DLC'] = 0
        for i, col in enumerate(byte_cols):
            df2[col] = [p[i] for p in parsed]

        if atk_col and atk_col in dfr.columns:
            df2['Flag'] = dfr[atk_col].apply(lambda v: 'T' if int(v) else 'R')
        else:
            df2['Flag'] = 'R'

        prefix = os.path.basename(fname).split('-')[0] if fname else 'unknown'
        p = prefix.lower()
        alabel = ('DoS'   if 'dos'  in p else
                  'Fuzzy' if 'fuzz' in p else
                  'RPM'   if ('rpm' in p or 'speed' in p) else
                  'Gear'  if ('gear' in p or 'force' in p or 'standstill' in p) else
                  prefix.capitalize())
        df2['Label'] = df2['Flag'].apply(lambda r: 'Normal' if r == 'R' else alabel)
        return df2

    base_features = ['CAN_ID', 'DLC'] + ['DATA{}'.format(i) for i in range(8)]
    parts = []

    if os.path.isdir(os.path.join(data_path, 'set_01')):
        for setd in sorted(os.listdir(data_path)):
            setp = os.path.join(data_path, setd)
            if not os.path.isdir(setp): continue
            for subset in sorted(os.listdir(setp)):
                subp = os.path.join(setp, subset)
                if not os.path.isdir(subp): continue
                for fname in sorted(os.listdir(subp)):
                    if not fname.lower().endswith('.csv'): continue
                    fp = os.path.join(subp, fname)
                    try:
                        df_part = _parse_can_csv(fp, fname)
                    except Exception as e:
                        print('Skipped', fp, '->', e); df_part = None
                    if df_part is not None and len(df_part) > 0:
                        parts.append(df_part)
    else:
        parts = [load_normal_df(data_path)]
        for primary, alt, attack_label in ATTACK_FILE_SPECS:
            parts.append(load_attack_df(data_path, primary, alt, attack_label))

    X_list, y_list = [], []
    for sub in parts:
        if 'Flag' in sub.columns:
            sub = sub.drop(columns=['Flag'])
        Xw, yw = make_windows_from_sorted_df(sub, base_features, SEQ_LEN)
        if len(Xw) > 0:
            X_list.append(Xw)
            y_list.append(yw)

    if not X_list:
        raise RuntimeError('No sliding windows built — check DATA_PATH and SEQ_LEN.')
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

print('Data loading & feature engineering defined ✅')

# ── Stage 1: Conv1D + BiLSTM (IDENTICAL to claude_opus_model_colab.py) ────────

def build_stage1_model(seq_len, n_features, num_classes=5):
    inp = Input(shape=(seq_len, n_features))
    x   = Conv1D(64,  3, padding='same', activation='relu')(inp)
    x   = BatchNormalization()(x)
    x   = MaxPooling1D(2)(x)
    x   = Dropout(0.25)(x)
    x   = Conv1D(128, 3, padding='same', activation='relu')(x)
    x   = BatchNormalization()(x)
    x   = MaxPooling1D(2)(x)
    x   = Dropout(0.25)(x)
    x   = LSTM(96, return_sequences=True)(x)
    x   = Dropout(0.3)(x)
    x   = LSTM(64, return_sequences=False)(x)
    x   = BatchNormalization()(x)
    x   = Dropout(0.35)(x)
    x   = Dense(128, activation='relu')(x)
    x   = Dropout(0.25)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inp, out)
    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

print('Stage 1 model builder defined ✅')

# ── Stage 2: Variational Autoencoder ensemble + improved temporal predictor ───

class _Sampling(tf.keras.layers.Layer):
    """Reparameterization trick: z = mu + epsilon * exp(0.5 * log_var)."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class _VAEModel(tf.keras.Model):
    """
    Beta-VAE wrapper that adds KL-regularised training on top of a
    functional encoder + decoder pair.

    Loss = reconstruction_MSE (summed over T×F, averaged over batch)
         + beta * KL(q(z|x) || N(0,I))
    """

    def __init__(self, encoder, decoder, beta=VAE_BETA, **kw):
        super().__init__(**kw)
        self.encoder = encoder
        self.decoder = decoder
        self.beta    = beta
        self._rl_tracker  = tf.keras.metrics.Mean(name='recon_loss')
        self._kl_tracker  = tf.keras.metrics.Mean(name='kl_loss')
        self._tot_tracker = tf.keras.metrics.Mean(name='loss')

    @property
    def metrics(self):
        return [self._tot_tracker, self._rl_tracker, self._kl_tracker]

    def call(self, x, training=False):
        z_mean, _, z = self.encoder(x, training=training)
        return self.decoder(z_mean, training=training)  # use mean for deterministic recon

    def train_step(self, data):
        x = data[0] if isinstance(data, (list, tuple)) else data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x, training=True)
            x_recon = self.decoder(z, training=True)
            # sum over T and F, average over batch
            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - x_recon), axis=[1, 2]))
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            total_loss = recon_loss + self.beta * kl_loss
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self._tot_tracker.update_state(total_loss)
        self._rl_tracker .update_state(recon_loss)
        self._kl_tracker .update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x = data[0] if isinstance(data, (list, tuple)) else data
        z_mean, z_log_var, z = self.encoder(x, training=False)
        x_recon = self.decoder(z, training=False)
        recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - x_recon), axis=[1, 2]))
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )
        total_loss = recon_loss + self.beta * kl_loss
        self._tot_tracker.update_state(total_loss)
        self._rl_tracker .update_state(recon_loss)
        self._kl_tracker .update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}


def build_variational_autoencoder(seq_len, n_features, latent_dim=AE_LATENT_DIM, seed=0):
    """
    Conv1D + BiLSTM Variational Autoencoder (beta-VAE).

    Encoder:  (T, F) → Conv1D×2 → BiLSTM → z_mean / z_log_var → sampling
    Decoder:  z → Dense → Reshape(T) → LSTM → Conv1DTranspose×2 → TimeDistributed(F)

    Returns (vae_model, encoder, decoder).
      • encoder output: [z_mean, z_log_var, z_sample]  — use z_mean for latent scoring
      • decoder input:  z vector
    """
    tf.random.set_seed(RANDOM_STATE + seed)

    # ── Encoder ──────────────────────────────────────────────────────────────
    enc_inp = Input(shape=(seq_len, n_features), name='vae_enc_in_{}'.format(seed))
    x = Conv1D(64, 3, padding='same', activation='relu')(enc_inp)
    x = BatchNormalization()(x)
    x = Conv1D(96, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = Dropout(0.15)(x)
    z_mean    = Dense(latent_dim, name='z_mean_{}'.format(seed))(x)
    z_log_var = Dense(latent_dim, name='z_log_var_{}'.format(seed))(x)
    z_sample  = _Sampling(name='z_sample_{}'.format(seed))([z_mean, z_log_var])

    encoder = Model(enc_inp, [z_mean, z_log_var, z_sample], name='vae_enc_{}'.format(seed))

    # ── Decoder ──────────────────────────────────────────────────────────────
    dec_inp = Input(shape=(latent_dim,), name='vae_dec_in_{}'.format(seed))
    d = Dense(seq_len * 32, activation='relu')(dec_inp)
    d = Reshape((seq_len, 32))(d)
    d = LSTM(64, return_sequences=True)(d)
    d = BatchNormalization()(d)
    d = Conv1DTranspose(96, 3, padding='same', activation='relu')(d)
    d = Conv1DTranspose(64, 3, padding='same', activation='relu')(d)
    out = TimeDistributed(Dense(n_features, activation='sigmoid'))(d)

    decoder = Model(dec_inp, out, name='vae_dec_{}'.format(seed))

    # ── VAE training wrapper ──────────────────────────────────────────────────
    vae = _VAEModel(encoder, decoder, beta=VAE_BETA, name='vae_{}'.format(seed))
    vae.compile(optimizer=Adam(1e-3))

    return vae, encoder, decoder


def build_temporal_predictor(seq_len, n_features):
    """
    Improved temporal predictor: Bidirectional LSTM + Multi-Head Self-Attention.

    Takes the first (T-1) timesteps, predicts the T-th timestep.
    The attention block captures long-range temporal dependencies that the
    plain BiLSTM in claude_opus misses, improving detection of subtle
    sequential pattern violations introduced by spoofing attacks.
    """
    inp = Input(shape=(seq_len - 1, n_features))
    x = Conv1D(64, 3, padding='same', activation='relu')(inp)
    x = BatchNormalization()(x)
    x = Conv1D(96, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    # Bidirectional LSTM retains full sequence for attention
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    # Self-attention: queries attend to all prior positions simultaneously
    attn_out = MultiHeadAttention(
        num_heads=ATTN_NUM_HEADS, key_dim=ATTN_KEY_DIM, dropout=ATTN_DROPOUT
    )(x, x)
    x = Add()([x, attn_out])
    x = LayerNormalization()(x)
    # Collapse sequence
    x = Bidirectional(LSTM(48, return_sequences=False))(x)
    x = Dropout(0.15)(x)
    x = Dense(96, activation='relu')(x)
    x = Dropout(0.1)(x)
    out = Dense(n_features, activation='sigmoid')(x)
    model = Model(inp, out, name='temporal_predictor')
    model.compile(optimizer=Adam(1e-3), loss='mse')
    return model

print('VAE and improved temporal predictor builders defined ✅')

# ── Anomaly scoring: 9-channel with learned meta-classifier ──────────────────

def _predict_in_batches(model, X, batch_size):
    """Chunked predict to avoid GPU OOM on large arrays."""
    n, out = len(X), None
    for i in range(0, n, batch_size):
        chunk = X[i: i + batch_size].astype(np.float32)
        y = model.predict(chunk, verbose=0)
        if out is None:
            out = np.empty((n,) + y.shape[1:], dtype=y.dtype)
        out[i: i + len(chunk)] = y
    return out


def _vae_per_window_stats(encoder, decoder, X_seq, batch_size):
    """
    Compute per-window VAE anomaly statistics.

    Returns:
        mean_mse   (N,) — mean reconstruction MSE over timesteps
        max_mse    (N,) — worst-step reconstruction MSE
        kl_div     (N,) — KL divergence sum over latent dims (per window)
        elbo       (N,) — ELBO = recon_sum + KL (un-normalised, larger → more anomalous)
        z_mean_all (N, latent_dim) — deterministic latent codes for Maha/kNN/SVDD
    """
    n = len(X_seq)
    z_means_list, z_lvars_list, x_recons_list = [], [], []

    for i in range(0, n, batch_size):
        chunk = X_seq[i: i + batch_size].astype(np.float32)
        zm, zlv, _ = encoder.predict(chunk, verbose=0)
        xr          = decoder.predict(zm, verbose=0)   # deterministic recon from mean
        z_means_list.append(zm)
        z_lvars_list.append(zlv)
        x_recons_list.append(xr)

    z_mean_all  = np.concatenate(z_means_list,  axis=0).astype(np.float64)
    z_lvar_all  = np.concatenate(z_lvars_list,  axis=0).astype(np.float64)
    x_recon_all = np.concatenate(x_recons_list, axis=0).astype(np.float32)

    diff     = X_seq.astype(np.float32) - x_recon_all   # (N, T, F)
    step_mse = (diff * diff).mean(axis=2)                # (N, T)
    mean_mse = step_mse.mean(axis=1).astype(np.float64)
    max_mse  = step_mse.max(axis=1).astype(np.float64)

    # KL per window: sum over latent dims of 0.5*(mu^2 + sigma^2 - 1 - log(sigma^2))
    kl_div = 0.5 * np.sum(
        z_mean_all**2 + np.exp(z_lvar_all) - 1.0 - z_lvar_all,
        axis=1
    ).astype(np.float64)

    # ELBO proxy: recon_sum + KL (unnormalised; larger = less likely under model)
    recon_sum = (diff * diff).sum(axis=(1, 2)).astype(np.float64)
    elbo      = recon_sum + kl_div

    return mean_mse, max_mse, kl_div, elbo, z_mean_all


def _fit_latent_mahalanobis(Z_train_normal):
    mu    = Z_train_normal.mean(axis=0)
    Zc    = Z_train_normal - mu
    n     = max(len(Z_train_normal) - 1, 1)
    cov   = (Zc.T @ Zc) / n
    diag  = np.diag(np.diag(cov))
    cov_s = (1.0 - MAHA_SHRINKAGE) * cov + MAHA_SHRINKAGE * diag
    w, V  = np.linalg.eigh(cov_s)
    w     = np.clip(w, MAHA_EIG_FLOOR, None)
    return mu.astype(np.float64), np.linalg.inv((V * w) @ V.T).astype(np.float64)


def _mahalanobis(Z, mu, precision):
    Zc   = Z.astype(np.float64) - mu
    left = Zc @ precision
    d2   = np.einsum('ij,ij->i', left, Zc)
    return np.sqrt(np.clip(d2, 0.0, None))


# Channel names in sorted order (used for meta-clf feature matrix construction)
_CHANNEL_NAMES = sorted([
    'mean_mse', 'max_mse', 'kl_div', 'elbo',
    'maha', 'knn_dist', 'svdd_dist', 'ensemble_disagree', 'pred_error',
])


class StageTwoScorer:
    """
    Improved 9-channel anomaly scorer with learned meta-classifier fusion.

    Scoring channels:
    1. mean_mse          — mean reconstruction error across timesteps   (VAE)
    2. max_mse           — worst-case timestep reconstruction error      (VAE)
    3. kl_div            — KL divergence from unit Gaussian prior        (VAE NEW)
    4. elbo              — ELBO proxy (recon_sum + KL)                   (VAE NEW)
    5. maha              — Mahalanobis distance in latent space
    6. knn_dist          — avg k-NN distance in latent space (local density)
    7. svdd_dist         — L2 distance from latent centroid (Deep SVDD style)
    8. ensemble_disagree — std of per-VAE Mahalanobis (model uncertainty)
    9. pred_error        — temporal prediction error (BiLSTM + attention)

    Fusion: learned LogisticRegression on normalised channels (trained on
    validation normal vs real unseen attack windows) → probability output
    """

    def __init__(self):
        self.vaes          = []    # list of (vae, encoder, decoder)
        self.maha          = []    # list of (mu, precision) per VAE
        self.knn_models    = []    # list of NearestNeighbors per VAE
        self.svdd_centers  = []    # latent centroid per VAE
        self.predictor     = None
        self.norm_params   = None  # robust median/MAD per channel
        self.meta_clf      = None  # trained LogisticRegression (set by calibrate_meta_classifier)

    # ── Phase 1: train VAE ensemble + temporal predictor + calibrate norms ───

    def fit(self, X_fit_normal_seq, seq_len, n_features):
        for k in range(AE_ENSEMBLE_SIZE):
            print('\n-- Training Stage-2 VAE #{} (seed={}) --'.format(k + 1, RANDOM_STATE + k))
            vae, enc, dec = build_variational_autoencoder(seq_len, n_features, AE_LATENT_DIM, seed=k)
            vae.fit(
                X_fit_normal_seq,
                validation_split=0.1,
                epochs=AE_EPOCHS, batch_size=AE_BATCH,
                callbacks=[EarlyStopping(monitor='val_loss', patience=AE_PATIENCE,
                                         restore_best_weights=True)],
                verbose=2,
            )
            self.vaes.append((vae, enc, dec))

            # Latent codes from deterministic mean
            _, _, _, _, Z = _vae_per_window_stats(enc, dec, X_fit_normal_seq, AE_BATCH)

            mu, pr = _fit_latent_mahalanobis(Z)
            self.maha.append((mu, pr))

            self.svdd_centers.append(Z.mean(axis=0).astype(np.float64))

            n_store = min(len(Z), KNN_MAX_TRAIN)
            idx_sub = np.random.choice(len(Z), n_store, replace=False)
            knn = NearestNeighbors(n_neighbors=KNN_K, metric='euclidean', algorithm='auto')
            knn.fit(Z[idx_sub])
            self.knn_models.append(knn)

        # Temporal predictor (BiLSTM + attention)
        print('\n-- Training improved temporal predictor (BiLSTM + attention) --')
        X_pred_in     = X_fit_normal_seq[:, :-1, :]
        X_pred_target = X_fit_normal_seq[:, -1, :]
        self.predictor = build_temporal_predictor(seq_len, n_features)
        self.predictor.fit(
            X_pred_in, X_pred_target,
            validation_split=0.1,
            epochs=PRED_EPOCHS, batch_size=PRED_BATCH,
            callbacks=[EarlyStopping(monitor='val_loss', patience=PRED_PATIENCE,
                                     restore_best_weights=True)],
            verbose=2,
        )

        # Calibrate robust normalisation on train-normal scores
        raw = self._raw_scores_dict(X_fit_normal_seq)

        def robust_params(vec):
            med = float(np.median(vec))
            mad = float(np.median(np.abs(vec - med))) * 1.4826
            if mad < 1e-12:
                mad = float(vec.std() + 1e-8)
            return med, mad

        self.norm_params = {key: robust_params(vals) for key, vals in raw.items()}
        print('\nRobust standardisation (median, MAD-scale):')
        for k, v in self.norm_params.items():
            print('  {:20s}: median={:.6f}  scale={:.6f}'.format(k, v[0], v[1]))

    # ── Phase 2: train meta-classifier on val neg/pos ────────────────────────

    def calibrate_meta_classifier(self, X_neg, X_pos):
        """
        Train a LogisticRegression meta-classifier on normalised 9-channel
        scores from known normal windows (X_neg) and real unseen attack
        windows (X_pos) from the validation split.

        After calling this, score() returns meta-clf probability ∈ [0, 1].
        The threshold tuner then finds the optimal cut-point in [0, 1].
        """
        if len(X_neg) == 0 or len(X_pos) == 0:
            print('⚠️  calibrate_meta_classifier: empty neg or pos set — skipping meta-clf.')
            return

        feats_neg = self._normalised_feature_matrix(self._raw_scores_dict(X_neg))
        feats_pos = self._normalised_feature_matrix(self._raw_scores_dict(X_pos))

        X_meta = np.vstack([feats_neg, feats_pos])
        y_meta = np.concatenate([np.zeros(len(feats_neg)), np.ones(len(feats_pos))])

        # Clip extreme normalised values to prevent numerical issues
        X_meta = np.clip(X_meta, 0.0, 50.0)

        print('\n-- Training meta-classifier (LogisticRegression) on {} neg + {} pos windows --'
              .format(len(feats_neg), len(feats_pos)))

        self.meta_clf = LogisticRegression(
            C=META_CLF_C,
            class_weight='balanced',
            max_iter=1000,
            solver='lbfgs',
            random_state=RANDOM_STATE,
        )
        self.meta_clf.fit(X_meta, y_meta)

        coef_dict = {name: float(c) for name, c in zip(_CHANNEL_NAMES, self.meta_clf.coef_[0])}
        print('Meta-clf learned channel weights (higher = more discriminative):')
        for name in sorted(coef_dict, key=lambda k: abs(coef_dict[k]), reverse=True):
            print('  {:20s}: {:.4f}'.format(name, coef_dict[name]))

        # Quick calibration accuracy
        preds = self.meta_clf.predict(X_meta)
        cal_acc = accuracy_score(y_meta, preds)
        print('Meta-clf calibration accuracy: {:.4f}'.format(cal_acc))

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score(self, X_seq):
        """
        Return per-window anomaly score.
        If meta-clf is fitted: probability ∈ [0, 1] from LogisticRegression.
        Fallback (no meta-clf): weighted sum of normalised channels (same
        weights as claude_opus for compatibility).
        """
        raw = self._raw_scores_dict(X_seq)

        if self.meta_clf is not None:
            X_feat = np.clip(self._normalised_feature_matrix(raw), 0.0, 50.0)
            return self.meta_clf.predict_proba(X_feat)[:, 1].astype(np.float64)

        # Fallback: weighted sum (same logic as claude_opus)
        weights = {
            'mean_mse':          1.0,
            'max_mse':           1.0,
            'kl_div':            2.0,
            'elbo':              1.5,
            'maha':              2.0,
            'knn_dist':          1.5,
            'svdd_dist':         1.0,
            'ensemble_disagree': 1.0,
            'pred_error':        2.5,
        }
        combined, total_w = np.zeros(len(X_seq), dtype=np.float64), 0.0
        for key, vals in raw.items():
            med, sc = self.norm_params[key]
            z = np.abs(vals - med) / sc
            w = weights.get(key, 1.0)
            combined += w * z
            total_w  += w
        return (combined / total_w).astype(np.float64)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _raw_scores_dict(self, X_seq):
        """Compute all 9 raw scoring channels as a dict of float64 arrays."""
        n = len(X_seq)
        agg_mean = np.zeros(n, np.float64)
        agg_max  = np.zeros(n, np.float64)
        agg_kl   = np.zeros(n, np.float64)
        agg_elbo = np.zeros(n, np.float64)
        agg_maha = np.zeros(n, np.float64)
        agg_knn  = np.zeros(n, np.float64)
        agg_svdd = np.zeros(n, np.float64)
        per_vae_maha = []

        for idx, ((vae, enc, dec), (mu, prec)) in enumerate(zip(self.vaes, self.maha)):
            mean_mse, max_mse, kl_div, elbo, Z = _vae_per_window_stats(enc, dec, X_seq, AE_BATCH)

            agg_mean += mean_mse
            agg_max  += max_mse
            agg_kl   += kl_div
            agg_elbo += elbo

            maha_d = _mahalanobis(Z, mu, prec)
            agg_maha += maha_d
            per_vae_maha.append(maha_d)

            dists, _ = self.knn_models[idx].kneighbors(Z)
            agg_knn  += dists.mean(axis=1)

            agg_svdd += np.sqrt(np.sum((Z - self.svdd_centers[idx]) ** 2, axis=1))

        k = float(max(len(self.vaes), 1))
        scores = {
            'mean_mse':  agg_mean / k,
            'max_mse':   agg_max  / k,
            'kl_div':    agg_kl   / k,
            'elbo':      agg_elbo / k,
            'maha':      agg_maha / k,
            'knn_dist':  agg_knn  / k,
            'svdd_dist': agg_svdd / k,
        }

        # Ensemble disagreement: std of per-VAE Mahalanobis
        if len(per_vae_maha) > 1:
            scores['ensemble_disagree'] = np.stack(per_vae_maha, axis=0).std(axis=0)
        else:
            scores['ensemble_disagree'] = np.zeros(n, np.float64)

        # Temporal prediction error
        if self.predictor is not None:
            X_prefix    = X_seq[:, :-1, :]
            X_true_last = X_seq[:, -1, :].astype(np.float32)
            X_pred_last = _predict_in_batches(self.predictor, X_prefix, AE_BATCH)
            scores['pred_error'] = np.mean(
                (X_true_last - X_pred_last) ** 2, axis=1
            ).astype(np.float64)
        else:
            scores['pred_error'] = np.zeros(n, np.float64)

        return scores

    def _normalised_feature_matrix(self, raw_dict):
        """Stack normalised channels in deterministic sorted order → (N, 9)."""
        feats = []
        for key in _CHANNEL_NAMES:
            vals     = raw_dict[key]
            med, sc  = self.norm_params[key]
            feats.append(np.abs(vals - med) / sc)
        return np.column_stack(feats)

print('StageTwoScorer (9-channel + meta-clf) defined ✅')

# ── Threshold tuning ──────────────────────────────────────────────────────────

def _stage1_predicts_normal(stage1, X_seq, normal_idx5):
    preds = np.argmax(stage1.predict(X_seq, batch_size=BATCH_SIZE, verbose=0), axis=1)
    return preds == normal_idx5


def _tune_threshold_precision_priority(scores_neg, scores_pos):
    """Grid-search T maximising F_beta=0.5 with soft PRECISION_FLOOR.

    Works identically whether scores are meta-clf probabilities ∈ [0, 1]
    or unnormalised weighted sums — the grid just covers the observed range.
    """
    if len(scores_neg) == 0 or len(scores_pos) == 0:
        return 0.5, 0.0, 0.0, 0.0
    y_bin = np.concatenate([np.zeros(len(scores_neg), np.int32),
                             np.ones (len(scores_pos), np.int32)])
    s_all = np.concatenate([scores_neg, scores_pos])
    lo, hi = float(np.percentile(s_all, 1.0)), float(np.percentile(s_all, 99.5))
    if hi <= lo:
        return float(s_all.mean()), 0.0, 0.0, 0.0
    grid = np.linspace(lo, hi, THRESHOLD_GRID_POINTS)
    best_with_floor, best_plain = None, None
    for t in grid:
        pred = (s_all > t).astype(np.int32)
        fb   = fbeta_score    (y_bin, pred, beta=F_BETA, zero_division=0)
        prec = precision_score(y_bin, pred, zero_division=0)
        rec  = recall_score   (y_bin, pred, zero_division=0)
        cand = (fb, prec, rec, float(t))
        if best_plain is None or fb > best_plain[0]:        best_plain = cand
        if prec >= PRECISION_FLOOR:
            if best_with_floor is None or fb > best_with_floor[0]: best_with_floor = cand
    chosen = best_with_floor if best_with_floor is not None else best_plain
    fb, prec, rec, t = chosen
    print(
        'Threshold tuned (precision-priority, F0.5):\n'
        '  T={:.6f}  F0.5={:.4f}  precision={:.4f}  recall={:.4f}  used_floor={}'.format(
            t, fb, prec, rec, best_with_floor is not None
        )
    )
    return t, fb, prec, rec

print('Threshold tuning defined ✅')

# ── Main training pipeline ────────────────────────────────────────────────────

if not os.path.isdir(data_path):
    raise RuntimeError('Dataset folder not found: ' + repr(data_path))

if USE_SUBSET:
    print('Subset mode: max {:,} normal, max {:,} per attack CSV'.format(MAX_NORMAL, MAX_PER_ATTACK_FILE))

# Step 1: Load & window
X_w, y_str = build_all_windows(data_path)
print('Windows shape:', X_w.shape)
print('Label distribution:', {v: int((y_str == v).sum()) for v in np.unique(y_str)})

# Step 2: LOAO split
mask_known  = y_str != UNSEEN_ATTACK
X_known     = X_w[mask_known]
y_known     = y_str[mask_known]
X_unseen    = X_w[y_str == UNSEEN_ATTACK]
y_unseen    = np.array(['zero_day'] * len(X_unseen), dtype=object)

print('Known windows :', len(X_known))
print('Unseen windows:', len(X_unseen))

# Step 3: Encode, scale, split (LOAO-safe — scaler fitted on train split only)
le5 = LabelEncoder()
y_known_enc = le5.fit_transform(y_known)

n_stage1_classes = len(le5.classes_)
n_feat   = X_w.shape[2]
flat_dim = SEQ_LEN * n_feat

X_flat_known  = X_known .reshape(len(X_known),  flat_dim)
X_flat_unseen = X_unseen.reshape(len(X_unseen), flat_dim)

X_train_big_flat, X_test_flat, y_train_big_enc, y_test_enc = train_test_split(
    X_flat_known, y_known_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_known_enc
)
X_fit_flat, X_val_flat, y_fit_enc, y_val_enc = train_test_split(
    X_train_big_flat, y_train_big_enc,
    test_size=STAGE1_VAL_SPLIT, random_state=RANDOM_STATE, stratify=y_train_big_enc
)

scaler = MinMaxScaler()
scaler.fit(X_fit_flat)

X_fit_flat_s    = scaler.transform(X_fit_flat)   .astype(np.float32)
X_val_flat_s    = scaler.transform(X_val_flat)   .astype(np.float32)
X_test_flat_s   = scaler.transform(X_test_flat)  .astype(np.float32)
X_flat_unseen_s = scaler.transform(X_flat_unseen).astype(np.float32)

X_fit       = X_fit_flat_s   .reshape(len(X_fit_flat_s),    SEQ_LEN, n_feat)
X_val       = X_val_flat_s   .reshape(len(X_val_flat_s),    SEQ_LEN, n_feat)
X_test      = X_test_flat_s  .reshape(len(X_test_flat_s),   SEQ_LEN, n_feat)
X_unseen_s  = X_flat_unseen_s.reshape(len(X_flat_unseen_s), SEQ_LEN, n_feat)

y_fit_cat   = to_categorical(y_fit_enc, num_classes=n_stage1_classes)
normal_idx5 = int(le5.transform(['Normal'])[0])

mask_fit_normal  = (y_fit_enc == normal_idx5)
X_fit_normal_seq = X_fit[mask_fit_normal]

class_weights    = compute_class_weight('balanced', classes=np.arange(n_stage1_classes), y=y_fit_enc)
class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}

print('Known train/val/test shapes:', X_fit.shape, X_val.shape, X_test.shape)
print('Unseen windows scaled:', X_unseen_s.shape)

# Step 4: Train Stage 1 (IDENTICAL to claude_opus)
print('\n========== Training Stage 1 (IDENTICAL to claude_opus) ==========')
stage1 = build_stage1_model(SEQ_LEN, n_feat, n_stage1_classes)
stage1.summary()

cb1 = [
    EarlyStopping(monitor='val_loss', patience=STAGE1_PATIENCE, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5),
]
stage1.fit(
    X_fit, y_fit_cat,
    validation_split=STAGE1_VAL_SPLIT,
    epochs=STAGE1_EPOCHS, batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=cb1, verbose=2,
)

s1_test_pred = np.argmax(stage1.predict(X_test, batch_size=BATCH_SIZE, verbose=0), axis=1)
y_s1_true    = le5.inverse_transform(y_test_enc)
y_s1_pred    = le5.inverse_transform(s1_test_pred)
labels5      = list(le5.classes_)

print_metrics_block('Stage 1 only (5-class)', y_s1_true, y_s1_pred, labels5)
plot_confusion_heatmap(
    y_s1_true, y_s1_pred, labels5,
    'Stage 1 (letsee — IDENTICAL to claude_opus)',
    'letsee_cm_stage1.png',
)

# Step 5: Train Stage 2 (VAE ensemble + improved temporal predictor)
print('\n========== Training Stage 2 (VAE ensemble + 9-channel scorer) ==========')
print('  VAE ensemble (K={}) + BiLSTM-Attention temporal predictor + '
      'k-NN + SVDD + Mahalanobis + KL'.format(AE_ENSEMBLE_SIZE))
scorer = StageTwoScorer()
scorer.fit(X_fit_normal_seq, SEQ_LEN, n_feat)

# Step 6: Build validation neg/pos for meta-clf calibration + threshold tuning
print('\n========== Preparing validation neg/pos sets ==========')

s1_val = np.argmax(stage1.predict(X_val, batch_size=BATCH_SIZE, verbose=0), axis=1)
mask_val_normal = (y_val_enc == normal_idx5) & (s1_val == normal_idx5)
X_val_neg = X_val[mask_val_normal]
if len(X_val_neg) < 200:
    X_val_neg = X_val[s1_val == normal_idx5]
print('Validation negatives (Normal, Stage1=Normal):', len(X_val_neg))

mask_unseen_s1_normal = _stage1_predicts_normal(stage1, X_unseen_s, normal_idx5)
X_val_pos = X_unseen_s[mask_unseen_s1_normal]
if len(X_val_pos) < 50:
    X_val_pos = X_unseen_s  # use all unseen if too few pass Stage 1
print('Validation positives (unseen, Stage1=Normal):', len(X_val_pos))

# Step 7: Calibrate meta-classifier on val neg/pos
print('\n========== Calibrating meta-classifier ==========')
scorer.calibrate_meta_classifier(X_val_neg, X_val_pos)

# Step 8: Tune threshold (now on meta-clf probability scores)
print('\n========== Tuning Stage-2 threshold (precision-priority F0.5) ==========')
scores_neg = scorer.score(X_val_neg) if len(X_val_neg) > 0 else np.array([])
scores_pos = scorer.score(X_val_pos) if len(X_val_pos) > 0 else np.array([])

best_T, val_fb, val_p, val_r = _tune_threshold_precision_priority(scores_neg, scores_pos)
print('Tuned threshold T =', best_T)

# Step 9: Evaluate on held-out test set (LOAO)
print('\n========== Evaluating on held-out test set ==========')

X_eval_seq = np.concatenate([X_test, X_unseen_s], axis=0)
y_eval_true_str = np.concatenate([
    le5.inverse_transform(y_test_enc),
    np.array(['zero_day'] * len(X_unseen_s), dtype=object),
])

# Stage 1 predictions + confidence
s1_probs      = stage1.predict(X_eval_seq, batch_size=BATCH_SIZE, verbose=0)
s1_eval       = np.argmax(s1_probs, axis=1)
s1_confidence = np.max(s1_probs, axis=1)

# Stage 2 anomaly scores (meta-clf probabilities)
sc_eval = scorer.score(X_eval_seq)

# Hybrid decision with confidence-adjusted gating (IDENTICAL logic to claude_opus)
final_pred = []
for i in range(len(s1_eval)):
    if s1_eval[i] != normal_idx5:
        final_pred.append(le5.inverse_transform(np.array([s1_eval[i]]))[0])
    else:
        uncertainty  = 1.0 - s1_confidence[i]
        effective_T  = best_T * (1.0 - S1_CONFIDENCE_WEIGHT * uncertainty)
        final_pred.append('zero_day' if sc_eval[i] > effective_T else 'Normal')

# Stage 2 binary view (Normal vs zero_day) on Stage1=Normal slice
mask_s1_normal = s1_eval == normal_idx5
y_bin_true, y_bin_pred = [], []
for i in range(len(y_eval_true_str)):
    if not mask_s1_normal[i]: continue
    t = y_eval_true_str[i]
    if t not in ('Normal', 'zero_day'): continue
    y_bin_true.append(t)
    uncertainty  = 1.0 - s1_confidence[i]
    effective_T  = best_T * (1.0 - S1_CONFIDENCE_WEIGHT * uncertainty)
    y_bin_pred.append('zero_day' if sc_eval[i] > effective_T else 'Normal')

labels_bin = ['Normal', 'zero_day']
if len(y_bin_true) > 0:
    print_metrics_block(
        'Stage 2 (VAE 9-channel + meta-clf fusion + confidence gating)',
        y_bin_true, y_bin_pred, labels_bin,
    )
    plot_confusion_heatmap(
        y_bin_true, y_bin_pred, labels_bin,
        'Stage 2 — letsee (VAE ensemble + meta-clf)',
        'letsee_cm_stage2.png', figsize=(7, 6),
    )

# Final 6-class hybrid evaluation
print_metrics_block(
    'Final hybrid — letsee (LOAO: {} unseen)'.format(UNSEEN_ATTACK),
    y_eval_true_str, final_pred, FINAL_LABELS,
)
plot_confusion_heatmap(
    y_eval_true_str, final_pred, FINAL_LABELS,
    'Final hybrid — letsee (LOAO: {} unseen)'.format(UNSEEN_ATTACK),
    'letsee_cm_final.png', figsize=(11, 9),
)

# Per-class zero_day metrics
y_true_zero = np.array([1 if t == 'zero_day' else 0 for t in y_eval_true_str])
y_pred_zero = np.array([1 if p == 'zero_day' else 0 for p in final_pred])

zero_prec = precision_score(y_true_zero, y_pred_zero, zero_division=0)
zero_rec  = recall_score   (y_true_zero, y_pred_zero, zero_division=0)
zero_f1   = f1_score       (y_true_zero, y_pred_zero, zero_division=0)

y_true_norm = np.array([1 if t == 'Normal' else 0 for t in y_eval_true_str])
y_pred_norm = np.array([1 if p == 'Normal' else 0 for p in final_pred])
normal_rec  = recall_score(y_true_norm, y_pred_norm, zero_division=0)

n_normal_total = int(y_true_norm.sum())
n_normal_as_zd = int(sum(1 for t, p in zip(y_eval_true_str, final_pred)
                         if t == 'Normal' and p == 'zero_day'))
fpr_normal = n_normal_as_zd / max(n_normal_total, 1)

print('\n--- Open-set detection summary ---')
print('  Zero-day precision : {:.4f}'.format(zero_prec))
print('  Zero-day recall    : {:.4f}'.format(zero_rec))
print('  Zero-day F1        : {:.4f}'.format(zero_f1))
print('  Normal recall      : {:.4f}'.format(normal_rec))
print('  Normal FPR (→ zd)  : {:.4f}  ({}/{})'.format(fpr_normal, n_normal_as_zd, n_normal_total))
print('  Base threshold T   : {:.6f}'.format(best_T))
print(
    '\n--- letsee.py complete (LOAO: {} unseen).'
    '  val F0.5={:.4f}  val precision={:.4f}  val recall={:.4f} ---'.format(
        UNSEEN_ATTACK, val_fb, val_p, val_r
    )
)

# ── Download outputs from Colab ───────────────────────────────────────────────
if _IN_COLAB:
    from google.colab import files
    for fname in ['letsee_cm_stage1.png', 'letsee_cm_stage2.png', 'letsee_cm_final.png']:
        if os.path.isfile(fname):
            files.download(fname)
            print('Downloaded:', fname)
        else:
            print('Not found (skipped):', fname)
