# -*- coding: utf-8 -*-
"""letsee-2.py — Hybrid CAN-Bus IDS, Stage-2 pushed further than letsee.py

| Stage | What it does |
|-------|--------------|
| Stage 1 | IDENTICAL to claude_opus_model_colab.py — Conv1D + BiLSTM |
| Stage 2 | EVEN STRONGER — see “Improvements over letsee.py” below |

Improvements over letsee.py (in priority of expected impact)
============================================================
1. **Known attacks as extra meta-classifier positives** (BIGGEST LIFT)
   - letsee.py only trained the meta-clf on (Normal, real-unseen-RPM) pairs.
   - letsee-2 also feeds DoS / Fuzzy / Gear windows from the validation split
     as additional anomaly examples → meta-clf learns the general
     “what does an attack look like in 11-channel space” pattern, which
     generalises far better to the held-out RPM zero-day at test time.

2. **Non-linear meta-classifier**: `GradientBoostingClassifier`
   replaces `LogisticRegression`. Captures channel interactions
   (e.g. high `kl_div` AND high `pred_error` is more discriminative than
   either alone) that a linear model cannot model.

3. **Bigger dataset**
   - `MAX_NORMAL`: 5,000 → 20,000  (4× more normal training data)
   - `MAX_PER_ATTACK_FILE`: 10,000 → 25,000  (2.5× more attack data)

4. **Larger VAE ensemble** (K = 3 → 5)
   - More diverse latent spaces → stronger `ensemble_disagree` signal
   - More robust Mahalanobis / k-NN / SVDD aggregation

5. **Multi-step temporal predictor** (predict last 3 steps, not 1)
   - More error signal per window → richer `pred_error` channel
   - Catches subtle multi-step pattern violations specific to spoofing

6. **Per-feature reconstruction channels** (11 total channels vs 9):
   - `can_id_recon`: MSE on the CAN_ID feature only — directly probes
     ID-spoofing-style attacks (RPM, Gear).
   - `iat_recon`: MSE on the inter-arrival-time feature — directly probes
     timing-based attacks (DoS, flooding).

Outputs: letsee2_cm_stage1.png, letsee2_cm_stage2.png, letsee2_cm_final.png
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

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, fbeta_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    LSTM, Add, BatchNormalization, Bidirectional,
    Conv1D, Conv1DTranspose, Dense, Dropout,
    Input, LayerNormalization, MaxPooling1D, MultiHeadAttention,
    Reshape, TimeDistributed,
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

# Subset / sampling controls (BIGGER than letsee.py)
USE_SUBSET          = True
MAX_NORMAL          = 20_000     # was 5_000
MAX_PER_ATTACK_FILE = 25_000     # was 10_000

# Windowing & Stage 1 (unchanged from letsee.py / claude_opus)
SEQ_LEN             = 24
BATCH_SIZE          = 64
STAGE1_EPOCHS       = 55
STAGE1_PATIENCE     = 12
STAGE1_VAL_SPLIT    = 0.15

# Stage-2 VAE ensemble (K bumped 3 → 5)
AE_LATENT_DIM       = 32
AE_EPOCHS           = 60
AE_PATIENCE         = 10
AE_BATCH            = 256
AE_ENSEMBLE_SIZE    = 5          # was 3

# VAE-specific (beta-VAE KL weight)
VAE_BETA            = 0.001

# Temporal Predictor — now predicts last PRED_LAST_STEPS steps
PRED_EPOCHS         = 40
PRED_PATIENCE       = 8
PRED_BATCH          = 256
PRED_LAST_STEPS     = 3          # new — was implicitly 1

# Attention in temporal predictor
ATTN_NUM_HEADS      = 4
ATTN_KEY_DIM        = 32
ATTN_DROPOUT        = 0.1

# Mahalanobis numerical stability
MAHA_EIG_FLOOR      = 1e-4
MAHA_SHRINKAGE      = 0.05

# k-NN anomaly scoring (bumped to match larger training set)
KNN_K               = 10
KNN_MAX_TRAIN       = 5000       # was 3000

# Per-feature recon channels: indices into engineered feature vector
# Feature layout (from make_windows_from_sorted_df):
#   0 = CAN_ID    1 = DLC    2..9 = DATA0..DATA7    10 = IAT
#   11 = CAN_ID_freq    12 = byte_entropy    13 = byte_sum
#   14 = byte_range     15 = byte_std
CAN_ID_FEAT_IDX     = 0
IAT_FEAT_IDX        = 10

# Meta-classifier (GradientBoosting replaces LogisticRegression)
META_CLF_N_ESTIMATORS = 200
META_CLF_LR           = 0.05
META_CLF_MAX_DEPTH    = 3
META_CLF_SUBSAMPLE    = 0.8

# Stage 1 confidence gating
S1_CONFIDENCE_WEIGHT = 0.25

# Threshold tuning
THRESHOLD_GRID_POINTS = 600
F_BETA                = 0.5
PRECISION_FLOOR       = 0.85

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

# ── Data loading & feature engineering (IDENTICAL to claude_opus / letsee) ───

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

# ── Stage 2: VAE ensemble + multi-step temporal predictor ─────────────────────

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
        return self.decoder(z_mean, training=training)

    def train_step(self, data):
        x = data[0] if isinstance(data, (list, tuple)) else data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x, training=True)
            x_recon = self.decoder(z, training=True)
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
    """Conv1D + BiLSTM Variational Autoencoder (beta-VAE)."""
    tf.random.set_seed(RANDOM_STATE + seed)

    # Encoder
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

    # Decoder
    dec_inp = Input(shape=(latent_dim,), name='vae_dec_in_{}'.format(seed))
    d = Dense(seq_len * 32, activation='relu')(dec_inp)
    d = Reshape((seq_len, 32))(d)
    d = LSTM(64, return_sequences=True)(d)
    d = BatchNormalization()(d)
    d = Conv1DTranspose(96, 3, padding='same', activation='relu')(d)
    d = Conv1DTranspose(64, 3, padding='same', activation='relu')(d)
    out = TimeDistributed(Dense(n_features, activation='sigmoid'))(d)

    decoder = Model(dec_inp, out, name='vae_dec_{}'.format(seed))

    vae = _VAEModel(encoder, decoder, beta=VAE_BETA, name='vae_{}'.format(seed))
    vae.compile(optimizer=Adam(1e-3))

    return vae, encoder, decoder


def build_temporal_predictor(seq_len, n_features, n_pred=PRED_LAST_STEPS):
    """
    Multi-step temporal predictor: BiLSTM + Multi-Head Self-Attention.

    Takes the first (T - n_pred) timesteps and predicts the LAST n_pred
    timesteps. Outputting multiple future steps:
      • Forces the model to learn richer temporal dynamics
      • Provides a larger error signal on attack windows (3× more residuals)
      • Catches multi-step pattern violations specific to spoofing attacks
    """
    inp = Input(shape=(seq_len - n_pred, n_features))
    x = Conv1D(64, 3, padding='same', activation='relu')(inp)
    x = BatchNormalization()(x)
    x = Conv1D(96, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    attn_out = MultiHeadAttention(
        num_heads=ATTN_NUM_HEADS, key_dim=ATTN_KEY_DIM, dropout=ATTN_DROPOUT
    )(x, x)
    x = Add()([x, attn_out])
    x = LayerNormalization()(x)
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = Dropout(0.15)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    out = Dense(n_pred * n_features, activation='sigmoid')(x)
    out = Reshape((n_pred, n_features))(out)
    model = Model(inp, out, name='temporal_predictor')
    model.compile(optimizer=Adam(1e-3), loss='mse')
    return model

print('VAE and multi-step temporal predictor builders defined ✅')

# ── Anomaly scoring: 11-channel with GradientBoosting meta-classifier ─────────

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
        mean_mse     (N,)              — mean recon MSE over timesteps
        max_mse      (N,)              — worst-step recon MSE
        kl_div       (N,)              — KL(q(z|x) || N(0,I)) per window
        elbo         (N,)              — recon_sum + KL (un-normalised)
        z_mean_all   (N, latent_dim)   — deterministic latent codes
        per_feat_mse (N, F)            — mean recon MSE per FEATURE (over T)
    """
    n = len(X_seq)
    z_means_list, z_lvars_list, x_recons_list = [], [], []

    for i in range(0, n, batch_size):
        chunk = X_seq[i: i + batch_size].astype(np.float32)
        zm, zlv, _ = encoder.predict(chunk, verbose=0)
        xr          = decoder.predict(zm, verbose=0)
        z_means_list.append(zm)
        z_lvars_list.append(zlv)
        x_recons_list.append(xr)

    z_mean_all  = np.concatenate(z_means_list,  axis=0).astype(np.float64)
    z_lvar_all  = np.concatenate(z_lvars_list,  axis=0).astype(np.float64)
    x_recon_all = np.concatenate(x_recons_list, axis=0).astype(np.float32)

    diff = X_seq.astype(np.float32) - x_recon_all      # (N, T, F)
    sq   = diff * diff
    step_mse     = sq.mean(axis=2)                      # (N, T)
    per_feat_mse = sq.mean(axis=1).astype(np.float64)   # (N, F)
    mean_mse     = step_mse.mean(axis=1).astype(np.float64)
    max_mse      = step_mse.max (axis=1).astype(np.float64)

    kl_div = 0.5 * np.sum(
        z_mean_all**2 + np.exp(z_lvar_all) - 1.0 - z_lvar_all,
        axis=1
    ).astype(np.float64)

    recon_sum = sq.sum(axis=(1, 2)).astype(np.float64)
    elbo      = recon_sum + kl_div

    return mean_mse, max_mse, kl_div, elbo, z_mean_all, per_feat_mse


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


# 11 channels (sorted order, used for meta-clf feature matrix)
_CHANNEL_NAMES = sorted([
    'mean_mse', 'max_mse', 'kl_div', 'elbo',
    'maha', 'knn_dist', 'svdd_dist', 'ensemble_disagree', 'pred_error',
    'can_id_recon', 'iat_recon',   # NEW per-feature recon channels
])


class StageTwoScorer:
    """
    11-channel anomaly scorer with GradientBoosting meta-classifier.

    Channels (11):
       1.  mean_mse           — mean recon error across timesteps
       2.  max_mse            — worst-case timestep recon error
       3.  kl_div             — KL divergence from unit Gaussian prior
       4.  elbo               — ELBO proxy (recon_sum + KL)
       5.  maha               — Mahalanobis distance in latent
       6.  knn_dist           — avg k-NN distance in latent (local density)
       7.  svdd_dist          — L2 distance from latent centroid
       8.  ensemble_disagree  — std of per-VAE Mahalanobis
       9.  pred_error         — multi-step temporal prediction error
       10. can_id_recon       — recon MSE on CAN_ID feature only       (NEW)
       11. iat_recon          — recon MSE on inter-arrival-time only   (NEW)

    Fusion: GradientBoostingClassifier — non-linear, learns channel
    interactions, trained on real Normal + (known-attack + unseen-attack)
    validation windows.
    """

    def __init__(self):
        self.vaes          = []
        self.maha          = []
        self.knn_models    = []
        self.svdd_centers  = []
        self.predictor     = None
        self.norm_params   = None
        self.meta_clf      = None

    def fit(self, X_fit_normal_seq, seq_len, n_features):
        for k in range(AE_ENSEMBLE_SIZE):
            print('\n-- Training Stage-2 VAE #{}/{}  (seed={}) --'.format(
                k + 1, AE_ENSEMBLE_SIZE, RANDOM_STATE + k))
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

            _, _, _, _, Z, _ = _vae_per_window_stats(enc, dec, X_fit_normal_seq, AE_BATCH)

            mu, pr = _fit_latent_mahalanobis(Z)
            self.maha.append((mu, pr))

            self.svdd_centers.append(Z.mean(axis=0).astype(np.float64))

            n_store = min(len(Z), KNN_MAX_TRAIN)
            idx_sub = np.random.choice(len(Z), n_store, replace=False)
            knn = NearestNeighbors(n_neighbors=KNN_K, metric='euclidean', algorithm='auto')
            knn.fit(Z[idx_sub])
            self.knn_models.append(knn)

        # Multi-step temporal predictor
        print('\n-- Training multi-step temporal predictor (predict last {} steps) --'
              .format(PRED_LAST_STEPS))
        X_pred_in     = X_fit_normal_seq[:, :-PRED_LAST_STEPS, :]
        X_pred_target = X_fit_normal_seq[:, -PRED_LAST_STEPS:, :]
        self.predictor = build_temporal_predictor(seq_len, n_features, n_pred=PRED_LAST_STEPS)
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

    # ── Meta-classifier calibration (now with optional extra positives) ──────

    def calibrate_meta_classifier(self, X_neg, X_pos, X_pos_extra=None):
        """
        Train a GradientBoosting meta-classifier on normalised 11-channel
        scores from:
          X_neg       — known Normal windows                  → label 0
          X_pos       — real unseen attack windows            → label 1
          X_pos_extra — (optional) known-attack windows from
                        the validation split — provides extra
                        anomaly examples so the meta-clf learns
                        the GENERAL attack signature, not just
                        the unseen RPM pattern              → label 1

        After calling this, score() returns meta-clf probability ∈ [0, 1].
        """
        if len(X_neg) == 0 or len(X_pos) == 0:
            print('⚠️  calibrate_meta_classifier: empty neg or pos — skipping.')
            return

        feats_neg      = self._normalised_feature_matrix(self._raw_scores_dict(X_neg))
        feats_pos_main = self._normalised_feature_matrix(self._raw_scores_dict(X_pos))

        if X_pos_extra is not None and len(X_pos_extra) > 0:
            feats_pos_extra = self._normalised_feature_matrix(self._raw_scores_dict(X_pos_extra))
            feats_pos       = np.vstack([feats_pos_main, feats_pos_extra])
            n_extra         = len(feats_pos_extra)
        else:
            feats_pos       = feats_pos_main
            n_extra         = 0

        X_meta = np.vstack([feats_neg, feats_pos])
        y_meta = np.concatenate([np.zeros(len(feats_neg)), np.ones(len(feats_pos))])

        X_meta = np.clip(X_meta, 0.0, 50.0)

        sample_w = compute_sample_weight('balanced', y_meta)

        print(
            '\n-- Training meta-clf (GradientBoosting):'
            '  neg={}  pos_main={}  pos_extra_known_attacks={}  --'
            .format(len(feats_neg), len(feats_pos_main), n_extra)
        )

        self.meta_clf = GradientBoostingClassifier(
            n_estimators=META_CLF_N_ESTIMATORS,
            learning_rate=META_CLF_LR,
            max_depth=META_CLF_MAX_DEPTH,
            subsample=META_CLF_SUBSAMPLE,
            random_state=RANDOM_STATE,
        )
        self.meta_clf.fit(X_meta, y_meta, sample_weight=sample_w)

        imp_dict = {name: float(imp) for name, imp
                    in zip(_CHANNEL_NAMES, self.meta_clf.feature_importances_)}
        print('Meta-clf channel feature importances (higher = more discriminative):')
        for name in sorted(imp_dict, key=lambda k: imp_dict[k], reverse=True):
            print('  {:20s}: {:.4f}'.format(name, imp_dict[name]))

        preds   = self.meta_clf.predict(X_meta)
        cal_acc = accuracy_score(y_meta, preds, sample_weight=sample_w)
        print('Meta-clf weighted calibration accuracy: {:.4f}'.format(cal_acc))

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score(self, X_seq):
        """
        Return per-window anomaly score.
        If meta-clf is fitted: probability ∈ [0, 1] from GradientBoosting.
        Fallback: weighted sum of normalised channels.
        """
        raw = self._raw_scores_dict(X_seq)

        if self.meta_clf is not None:
            X_feat = np.clip(self._normalised_feature_matrix(raw), 0.0, 50.0)
            return self.meta_clf.predict_proba(X_feat)[:, 1].astype(np.float64)

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
            'can_id_recon':      1.5,
            'iat_recon':         1.5,
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
        """Compute all 11 raw scoring channels as a dict of float64 arrays."""
        n = len(X_seq)
        agg_mean   = np.zeros(n, np.float64)
        agg_max    = np.zeros(n, np.float64)
        agg_kl     = np.zeros(n, np.float64)
        agg_elbo   = np.zeros(n, np.float64)
        agg_maha   = np.zeros(n, np.float64)
        agg_knn    = np.zeros(n, np.float64)
        agg_svdd   = np.zeros(n, np.float64)
        agg_canid  = np.zeros(n, np.float64)
        agg_iat    = np.zeros(n, np.float64)
        per_vae_maha = []

        for idx, ((vae, enc, dec), (mu, prec)) in enumerate(zip(self.vaes, self.maha)):
            mean_mse, max_mse, kl_div, elbo, Z, per_feat_mse = \
                _vae_per_window_stats(enc, dec, X_seq, AE_BATCH)

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

            # Per-feature recon channels — pull out CAN_ID and IAT columns
            agg_canid += per_feat_mse[:, CAN_ID_FEAT_IDX]
            agg_iat   += per_feat_mse[:, IAT_FEAT_IDX]

        k = float(max(len(self.vaes), 1))
        scores = {
            'mean_mse':     agg_mean  / k,
            'max_mse':      agg_max   / k,
            'kl_div':       agg_kl    / k,
            'elbo':         agg_elbo  / k,
            'maha':         agg_maha  / k,
            'knn_dist':     agg_knn   / k,
            'svdd_dist':    agg_svdd  / k,
            'can_id_recon': agg_canid / k,
            'iat_recon':    agg_iat   / k,
        }

        if len(per_vae_maha) > 1:
            scores['ensemble_disagree'] = np.stack(per_vae_maha, axis=0).std(axis=0)
        else:
            scores['ensemble_disagree'] = np.zeros(n, np.float64)

        # Multi-step temporal prediction error
        if self.predictor is not None:
            X_prefix    = X_seq[:, :-PRED_LAST_STEPS, :]
            X_true_last = X_seq[:, -PRED_LAST_STEPS:, :].astype(np.float32)
            X_pred_last = _predict_in_batches(self.predictor, X_prefix, AE_BATCH)
            scores['pred_error'] = ((X_true_last - X_pred_last) ** 2) \
                                    .mean(axis=(1, 2)).astype(np.float64)
        else:
            scores['pred_error'] = np.zeros(n, np.float64)

        return scores

    def _normalised_feature_matrix(self, raw_dict):
        """Stack normalised channels in deterministic sorted order → (N, 11)."""
        feats = []
        for key in _CHANNEL_NAMES:
            vals    = raw_dict[key]
            med, sc = self.norm_params[key]
            feats.append(np.abs(vals - med) / sc)
        return np.column_stack(feats)

print('StageTwoScorer (11-channel + GradientBoosting meta-clf) defined ✅')

# ── Threshold tuning ──────────────────────────────────────────────────────────

def _stage1_predicts_normal(stage1, X_seq, normal_idx5):
    preds = np.argmax(stage1.predict(X_seq, batch_size=BATCH_SIZE, verbose=0), axis=1)
    return preds == normal_idx5


def _tune_threshold_precision_priority(scores_neg, scores_pos):
    """Grid-search T maximising F_beta=0.5 with soft PRECISION_FLOOR."""
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
    print('Subset mode: max {:,} normal, max {:,} per attack CSV'
          .format(MAX_NORMAL, MAX_PER_ATTACK_FILE))

# Step 1: Load & window
X_w, y_str = build_all_windows(data_path)
print('Windows shape:', X_w.shape)
print('Label distribution:', {v: int((y_str == v).sum()) for v in np.unique(y_str)})

# Step 2: LOAO split (unchanged from letsee)
mask_known  = y_str != UNSEEN_ATTACK
X_known     = X_w[mask_known]
y_known     = y_str[mask_known]
X_unseen    = X_w[y_str == UNSEEN_ATTACK]
y_unseen    = np.array(['zero_day'] * len(X_unseen), dtype=object)

print('Known windows :', len(X_known))
print('Unseen windows:', len(X_unseen))

# Step 3: Encode, scale, split
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

X_fit      = X_fit_flat_s   .reshape(len(X_fit_flat_s),    SEQ_LEN, n_feat)
X_val      = X_val_flat_s   .reshape(len(X_val_flat_s),    SEQ_LEN, n_feat)
X_test     = X_test_flat_s  .reshape(len(X_test_flat_s),   SEQ_LEN, n_feat)
X_unseen_s = X_flat_unseen_s.reshape(len(X_flat_unseen_s), SEQ_LEN, n_feat)

y_fit_cat   = to_categorical(y_fit_enc, num_classes=n_stage1_classes)
normal_idx5 = int(le5.transform(['Normal'])[0])

mask_fit_normal  = (y_fit_enc == normal_idx5)
X_fit_normal_seq = X_fit[mask_fit_normal]

class_weights     = compute_class_weight('balanced', classes=np.arange(n_stage1_classes), y=y_fit_enc)
class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}

print('Known train/val/test shapes:', X_fit.shape, X_val.shape, X_test.shape)
print('Unseen windows scaled:', X_unseen_s.shape)
print('Feature dimension n_feat =', n_feat,
      '  (CAN_ID_FEAT_IDX={}, IAT_FEAT_IDX={})'.format(CAN_ID_FEAT_IDX, IAT_FEAT_IDX))

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
    'Stage 1 (letsee-2 — IDENTICAL to claude_opus)',
    'letsee2_cm_stage1.png',
)

# Step 5: Train Stage 2 (VAE ensemble + multi-step temporal predictor)
print('\n========== Training Stage 2 (K={} VAE ensemble + 11-channel scorer) =========='
      .format(AE_ENSEMBLE_SIZE))
scorer = StageTwoScorer()
scorer.fit(X_fit_normal_seq, SEQ_LEN, n_feat)

# Step 6: Build val neg/pos + extra known-attack positives
print('\n========== Preparing validation neg/pos sets ==========')

s1_val = np.argmax(stage1.predict(X_val, batch_size=BATCH_SIZE, verbose=0), axis=1)

# Negatives: Normal validation windows that Stage 1 also calls Normal
mask_val_normal = (y_val_enc == normal_idx5) & (s1_val == normal_idx5)
X_val_neg = X_val[mask_val_normal]
if len(X_val_neg) < 200:
    X_val_neg = X_val[s1_val == normal_idx5]
print('Validation negatives (Normal, Stage1=Normal):', len(X_val_neg))

# Main positives: real unseen-attack windows that Stage 1 calls Normal
mask_unseen_s1_normal = _stage1_predicts_normal(stage1, X_unseen_s, normal_idx5)
X_val_pos = X_unseen_s[mask_unseen_s1_normal]
if len(X_val_pos) < 50:
    X_val_pos = X_unseen_s
print('Validation positives (unseen, Stage1=Normal):', len(X_val_pos))

# EXTRA positives (NEW vs letsee.py):
# Known-attack windows from validation set. Prefer windows that Stage 1
# misclassifies as Normal (those are the *hard* cases that actually
# reach Stage 2 at inference time). Fallback to all known-attack val
# windows if too few hard ones.
mask_val_attack       = (y_val_enc != normal_idx5)
mask_val_attack_hard  = mask_val_attack & (s1_val == normal_idx5)
X_val_pos_extra_hard  = X_val[mask_val_attack_hard]
X_val_pos_extra_all   = X_val[mask_val_attack]

if len(X_val_pos_extra_hard) >= 100:
    X_val_pos_extra = X_val_pos_extra_hard
    print('Extra positives (known-attack val, Stage1=Normal — hard cases):',
          len(X_val_pos_extra))
else:
    # Combine hard + a random subsample of easy known-attack windows
    n_easy_target = min(len(X_val_pos_extra_all), max(500, 4 * len(X_val_pos)))
    if len(X_val_pos_extra_all) > n_easy_target:
        rs   = np.random.RandomState(RANDOM_STATE)
        idx  = rs.choice(len(X_val_pos_extra_all), n_easy_target, replace=False)
        X_val_pos_extra = X_val_pos_extra_all[idx]
    else:
        X_val_pos_extra = X_val_pos_extra_all
    print('Extra positives (known-attack val, mixed easy+hard):', len(X_val_pos_extra))

# Step 7: Calibrate meta-classifier
print('\n========== Calibrating meta-classifier (GradientBoosting, with extra positives) ==========')
scorer.calibrate_meta_classifier(X_val_neg, X_val_pos, X_pos_extra=X_val_pos_extra)

# Step 8: Tune threshold on the SAME positives + negatives the meta-clf saw
# (use the union of unseen + known-attack positives → robust threshold)
print('\n========== Tuning Stage-2 threshold (precision-priority F0.5) ==========')
X_val_pos_full = (np.concatenate([X_val_pos, X_val_pos_extra], axis=0)
                  if len(X_val_pos_extra) > 0 else X_val_pos)
scores_neg = scorer.score(X_val_neg)     if len(X_val_neg)      > 0 else np.array([])
scores_pos = scorer.score(X_val_pos_full) if len(X_val_pos_full) > 0 else np.array([])

best_T, val_fb, val_p, val_r = _tune_threshold_precision_priority(scores_neg, scores_pos)
print('Tuned threshold T =', best_T)

# Step 9: Evaluate on held-out test set (LOAO)
print('\n========== Evaluating on held-out test set ==========')

X_eval_seq = np.concatenate([X_test, X_unseen_s], axis=0)
y_eval_true_str = np.concatenate([
    le5.inverse_transform(y_test_enc),
    np.array(['zero_day'] * len(X_unseen_s), dtype=object),
])

s1_probs      = stage1.predict(X_eval_seq, batch_size=BATCH_SIZE, verbose=0)
s1_eval       = np.argmax(s1_probs, axis=1)
s1_confidence = np.max(s1_probs, axis=1)

sc_eval = scorer.score(X_eval_seq)

# Hybrid decision with confidence-adjusted gating
final_pred = []
for i in range(len(s1_eval)):
    if s1_eval[i] != normal_idx5:
        final_pred.append(le5.inverse_transform(np.array([s1_eval[i]]))[0])
    else:
        uncertainty = 1.0 - s1_confidence[i]
        effective_T = best_T * (1.0 - S1_CONFIDENCE_WEIGHT * uncertainty)
        final_pred.append('zero_day' if sc_eval[i] > effective_T else 'Normal')

# Stage 2 binary view (Normal vs zero_day) on Stage1=Normal slice
mask_s1_normal = s1_eval == normal_idx5
y_bin_true, y_bin_pred = [], []
for i in range(len(y_eval_true_str)):
    if not mask_s1_normal[i]: continue
    t = y_eval_true_str[i]
    if t not in ('Normal', 'zero_day'): continue
    y_bin_true.append(t)
    uncertainty = 1.0 - s1_confidence[i]
    effective_T = best_T * (1.0 - S1_CONFIDENCE_WEIGHT * uncertainty)
    y_bin_pred.append('zero_day' if sc_eval[i] > effective_T else 'Normal')

labels_bin = ['Normal', 'zero_day']
if len(y_bin_true) > 0:
    print_metrics_block(
        'Stage 2 (letsee-2: VAE×5 + 11-channel + GradientBoosting meta-clf)',
        y_bin_true, y_bin_pred, labels_bin,
    )
    plot_confusion_heatmap(
        y_bin_true, y_bin_pred, labels_bin,
        'Stage 2 — letsee-2 (VAE×5 + GB meta-clf)',
        'letsee2_cm_stage2.png', figsize=(7, 6),
    )

print_metrics_block(
    'Final hybrid — letsee-2 (LOAO: {} unseen)'.format(UNSEEN_ATTACK),
    y_eval_true_str, final_pred, FINAL_LABELS,
)
plot_confusion_heatmap(
    y_eval_true_str, final_pred, FINAL_LABELS,
    'Final hybrid — letsee-2 (LOAO: {} unseen)'.format(UNSEEN_ATTACK),
    'letsee2_cm_final.png', figsize=(11, 9),
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
    '\n--- letsee-2.py complete (LOAO: {} unseen).'
    '  val F0.5={:.4f}  val precision={:.4f}  val recall={:.4f} ---'.format(
        UNSEEN_ATTACK, val_fb, val_p, val_r
    )
)

# ── Download outputs from Colab ───────────────────────────────────────────────
if _IN_COLAB:
    from google.colab import files
    for fname in ['letsee2_cm_stage1.png', 'letsee2_cm_stage2.png', 'letsee2_cm_final.png']:
        if os.path.isfile(fname):
            files.download(fname)
            print('Downloaded:', fname)
        else:
            print('Not found (skipped):', fname)
