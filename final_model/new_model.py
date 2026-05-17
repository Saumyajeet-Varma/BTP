# -*- coding: utf-8 -*-
"""new_model.py — Hybrid CAN-Bus IDS

Reference paper (Stage 2):
    Althunayyan, M.; Javed, A.; Rana, O. (2024).
    “A robust multi-stage intrusion detection system for in-vehicle network
     security using hierarchical federated learning.”
     Vehicular Communications, 49, 100837.
     https://doi.org/10.1016/j.vehcom.2024.100837
     (arXiv preprint: 2408.08433)

| Stage | What it does                                                       |
|-------|--------------------------------------------------------------------|
| Stage 1 | IDENTICAL to letsee.py / claude_opus_model_colab.py               |
|         | Conv1D + BiLSTM multi-class classifier on KNOWN attacks           |
| Stage 2 | Paper-faithful **LSTM-Autoencoder**, time_steps=1, 9 features     |
|         | (CAN_ID + 8 data bytes), threshold = μ + σ of train recon errors  |

Why this hybrid?
================
The paper's own Stage 1 is a tiny feed-forward ANN
(Input(9) → Dense(16, ReLU) → Dense(16, ReLU) → softmax(5)).
We deliberately swap that out for our stronger Conv1D + BiLSTM backbone
(letsee.py / claude_opus_model_colab.py) because:
  • Our backbone exploits temporal context (SEQ_LEN=24) and engineered
    features like inter-arrival-time and byte entropy.
  • This is the same Stage 1 used by all our previous experiments,
    so results are directly comparable.
We keep the paper's Stage 2 design verbatim (LSTM-AE + μ+σ threshold)
so the comparison isolates the effect of the SECOND-STAGE choice.

Stage-1 / Stage-2 bridge
========================
Stage 1 takes sequence windows of shape (SEQ_LEN, n_feat). The paper's
Stage 2 takes individual CAN frames of shape (1, 9). To bridge them:
  • For each window flagged by Stage 1, take the **last frame**
    (the message whose label defines the window).
  • Project that frame onto the paper's 9 raw features:
        [CAN_ID, DATA0, DATA1, ..., DATA7].
  • Apply a StandardScaler fit ONLY on train-normal last-frames (per
    paper's preprocessing choice).

LOAO evaluation (from letsee.py)
================================
Hold one attack out (default RPM). Train Stage 1 on Normal + 3 known
attacks. Stage 2 trains on train-Normal frames only. At test time,
held-out RPM windows are evaluated as 'zero_day'.

Outputs:
    new_model_cm_stage1.png
    new_model_cm_stage2.png
    new_model_cm_final.png
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

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    LSTM, BatchNormalization, Conv1D, Dense, Dropout,
    Input, MaxPooling1D, RepeatVector, TimeDistributed,
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

# Subset / sampling controls (same as letsee.py)
USE_SUBSET          = True
MAX_NORMAL          = 5_000
MAX_PER_ATTACK_FILE = 10_000

# Windowing & Stage 1 (IDENTICAL to letsee.py / claude_opus)
SEQ_LEN             = 24
BATCH_SIZE          = 64
STAGE1_EPOCHS       = 55
STAGE1_PATIENCE     = 12
STAGE1_VAL_SPLIT    = 0.15

# Stage 2 — paper's LSTM-Autoencoder hyperparameters
# Source: Vehicular Communications 49, 100837 — Table 5
PAPER_AE_TIME_STEPS = 1
PAPER_AE_EPOCHS     = 100
PAPER_AE_BATCH      = 64
PAPER_AE_DROPOUT    = 0.2
PAPER_AE_VAL_SPLIT  = 0.1   # used internally for early-stopping monitoring

# Paper uses 9 raw features per CAN message: CAN_ID + 8 data bytes
# In our engineered feature vector (16 features) these correspond to
# indices [0, 2, 3, 4, 5, 6, 7, 8, 9]:
#   idx 0   = CAN_ID
#   idx 1   = DLC                  (paper EXCLUDES)
#   idx 2-9 = DATA0..DATA7
#   idx 10+ = engineered (IAT, freq, entropy, ...)  (paper EXCLUDES)
PAPER_FEATURE_INDICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]
PAPER_N_FEATURES      = len(PAPER_FEATURE_INDICES)

# Attack file specs
ATTACK_FILE_SPECS = [
    ('DoS_dataset.csv',   'dos_attack.csv',   'DoS'),
    ('Fuzzy_dataset.csv', 'fuzzy_attack.csv', 'Fuzzy'),
    ('gear_dataset.csv',  'gear_spoofing.csv','Gear'),
    ('RPM_dataset.csv',   'rpm_spoofing.csv', 'RPM'),
]

# LOAO (Leave-One-Attack-Out) — IDENTICAL to letsee.py
UNSEEN_ATTACK = "RPM"
ALL_ATTACKS   = ["DoS", "Fuzzy", "Gear", "RPM"]
KNOWN_ATTACKS = [a for a in ALL_ATTACKS if a != UNSEEN_ATTACK]
FINAL_LABELS  = ['Normal'] + KNOWN_ATTACKS + ['zero_day']

print('Paper Stage-2 feature indices :', PAPER_FEATURE_INDICES,
      '  (CAN_ID + DATA0..DATA7,', PAPER_N_FEATURES, 'features)')
print("Known attacks                 :", KNOWN_ATTACKS)
print("Unseen attack (zero_day)      :", UNSEEN_ATTACK)
print("Final labels                  :", FINAL_LABELS)
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

# ── Data loading & feature engineering (IDENTICAL to letsee.py) ──────────────

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

# ── Stage 1: Conv1D + BiLSTM (IDENTICAL to letsee.py / claude_opus) ──────────

def build_stage1_model(seq_len, n_features, num_classes=4):
    """Stage 1 — copied byte-for-byte from letsee.py / claude_opus_model_colab.py.

    Multi-class classifier over KNOWN classes only:
        ['Normal', 'DoS', 'Fuzzy', 'Gear']  (RPM held out as zero_day)
    """
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

# ── Stage 2: LSTM-Autoencoder per Althunayyan et al. 2024 ────────────────────

def build_paper_lstm_autoencoder(time_steps=PAPER_AE_TIME_STEPS,
                                 n_features=PAPER_N_FEATURES,
                                 dropout=PAPER_AE_DROPOUT):
    """
    Faithful re-implementation of the LSTM-Autoencoder from
    Althunayyan, Javed & Rana (2024), Vehicular Communications 49, 100837.

    Architecture (Section 4.4, Table 5 of the paper):
        Input  : (time_steps=1, features=9)
        Encoder: LSTM(128, return_sequences=True,  activation='relu')
                 LSTM(64,  return_sequences=False, activation='relu')
        Bottleneck: RepeatVector(time_steps)
        Decoder: LSTM(64,  return_sequences=True, activation='relu')
                 LSTM(128, return_sequences=True, activation='relu')
        Output : TimeDistributed(Dense(n_features))

    Hyper-parameters: Adam optimiser, MSE loss, Dropout = 0.2,
    Epochs = 100, Batch size = 64.

    The model is trained on Normal traffic only. At inference, an input
    whose reconstruction MSE exceeds (mean + std) of the training-set
    reconstruction errors is flagged as 'zero_day'.
    """
    inp = Input(shape=(time_steps, n_features), name='paper_ae_in')

    # Encoder
    x = LSTM(128, activation='relu', return_sequences=True,  name='enc_lstm_128')(inp)
    x = Dropout(dropout)(x)
    x = LSTM(64,  activation='relu', return_sequences=False, name='enc_lstm_64')(x)
    x = Dropout(dropout)(x)

    # Bottleneck — repeat the encoded vector along time axis
    x = RepeatVector(time_steps, name='repeat_vec')(x)

    # Decoder
    x = LSTM(64,  activation='relu', return_sequences=True, name='dec_lstm_64')(x)
    x = Dropout(dropout)(x)
    x = LSTM(128, activation='relu', return_sequences=True, name='dec_lstm_128')(x)
    x = Dropout(dropout)(x)

    out = TimeDistributed(Dense(n_features), name='td_dense_out')(x)
    model = Model(inp, out, name='paper_lstm_autoencoder')
    model.compile(optimizer=Adam(), loss='mse')
    return model


def extract_paper_frames_from_windows(X_windows):
    """
    Take the **last frame** of each sequence window and project onto the
    paper's 9 raw features [CAN_ID, DATA0, ..., DATA7].

    Input  : (N, SEQ_LEN, n_feat)  — sequence windows (Stage 1 input space)
    Output : (N, 9)                — single-frame paper features per window

    Rationale: in our pipeline a window is labelled by its LAST frame,
    so that frame is exactly the CAN message classified by Stage 1.
    Feeding that frame to the paper's Stage 2 LSTM-AE preserves both
    the paper's per-message granularity and our windowed Stage 1.
    """
    last_frames = X_windows[:, -1, :]                # (N, n_feat)
    return last_frames[:, PAPER_FEATURE_INDICES]     # (N, 9)


def compute_per_sample_mse(true, recon):
    """Per-sample MSE over (time, feature) axes."""
    err = (true - recon) ** 2
    if err.ndim == 3:
        return err.mean(axis=(1, 2)).astype(np.float64)
    return err.mean(axis=1).astype(np.float64)


print('Paper Stage-2 LSTM-AE + frame extractor defined ✅')

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

# Step 2: LOAO split — identical to letsee.py
mask_known  = y_str != UNSEEN_ATTACK
X_known     = X_w[mask_known]
y_known     = y_str[mask_known]
X_unseen    = X_w[y_str == UNSEEN_ATTACK]
y_unseen    = np.array(['zero_day'] * len(X_unseen), dtype=object)

print('Known windows :', len(X_known))
print('Unseen windows:', len(X_unseen))

# Step 3: Encode labels (known classes only)
le5 = LabelEncoder()
y_known_enc = le5.fit_transform(y_known)

n_stage1_classes = len(le5.classes_)
n_feat   = X_w.shape[2]
flat_dim = SEQ_LEN * n_feat
print('Stage 1 classes ({}):'.format(n_stage1_classes), list(le5.classes_))

# Stratified train/test/val split on the 3D windows directly
X_train_big, X_test, y_train_big_enc, y_test_enc = train_test_split(
    X_known, y_known_enc, test_size=0.2,
    random_state=RANDOM_STATE, stratify=y_known_enc,
)
X_fit_raw, X_val_raw, y_fit_enc, y_val_enc = train_test_split(
    X_train_big, y_train_big_enc,
    test_size=STAGE1_VAL_SPLIT,
    random_state=RANDOM_STATE, stratify=y_train_big_enc,
)

# ── Stage 1 scaling pipeline (MinMaxScaler on flat windows, as in letsee) ────
X_fit_flat   = X_fit_raw .reshape(len(X_fit_raw),  flat_dim)
X_val_flat   = X_val_raw .reshape(len(X_val_raw),  flat_dim)
X_test_flat  = X_test    .reshape(len(X_test),     flat_dim)
X_unseen_flat = X_unseen .reshape(len(X_unseen),   flat_dim)

scaler_s1 = MinMaxScaler()
scaler_s1.fit(X_fit_flat)

X_fit      = scaler_s1.transform(X_fit_flat)    .astype(np.float32).reshape(-1, SEQ_LEN, n_feat)
X_val      = scaler_s1.transform(X_val_flat)    .astype(np.float32).reshape(-1, SEQ_LEN, n_feat)
X_test_s   = scaler_s1.transform(X_test_flat)   .astype(np.float32).reshape(-1, SEQ_LEN, n_feat)
X_unseen_s = scaler_s1.transform(X_unseen_flat) .astype(np.float32).reshape(-1, SEQ_LEN, n_feat)

y_fit_cat   = to_categorical(y_fit_enc, num_classes=n_stage1_classes)
normal_idx5 = int(le5.transform(['Normal'])[0])

class_weights     = compute_class_weight('balanced', classes=np.arange(n_stage1_classes), y=y_fit_enc)
class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}

print('Stage-1 train/val/test shapes :', X_fit.shape, X_val.shape, X_test_s.shape)
print('Unseen windows (scaled)       :', X_unseen_s.shape)

# Step 4: Train Stage 1
print('\n========== Training Stage 1 (Conv1D+BiLSTM — same as letsee.py) ==========')
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

s1_test_pred = np.argmax(stage1.predict(X_test_s, batch_size=BATCH_SIZE, verbose=0), axis=1)
y_s1_true    = le5.inverse_transform(y_test_enc)
y_s1_pred    = le5.inverse_transform(s1_test_pred)
labels5      = list(le5.classes_)

print_metrics_block('Stage 1 only ({}-class)'.format(n_stage1_classes),
                    y_s1_true, y_s1_pred, labels5)
plot_confusion_heatmap(
    y_s1_true, y_s1_pred, labels5,
    'Stage 1 (new_model — Conv1D+BiLSTM, LOAO)',
    'new_model_cm_stage1.png',
)

# ── Stage 2 pipeline (paper-faithful) ────────────────────────────────────────
# Extract last-frame paper features (CAN_ID + DATA0..DATA7) for every window
# using the UNSCALED window tensors. We then fit StandardScaler on
# train-Normal frames only (no leakage from val/test/unseen).

print('\n========== Preparing Stage 2 inputs (paper-faithful frame extraction) ==========')
fit_frames_all     = extract_paper_frames_from_windows(X_fit_raw)     # (Nfit,  9)
val_frames_all     = extract_paper_frames_from_windows(X_val_raw)     # (Nval,  9)
test_frames_all    = extract_paper_frames_from_windows(X_test)        # (Ntest, 9)
unseen_frames_all  = extract_paper_frames_from_windows(X_unseen)      # (Nuns,  9)

mask_fit_normal      = (y_fit_enc == normal_idx5)
fit_frames_normal    = fit_frames_all[mask_fit_normal]
print('Stage-2 train-Normal frames (9 features):', fit_frames_normal.shape)

scaler_s2 = StandardScaler()                  # paper uses StandardScaler
scaler_s2.fit(fit_frames_normal)

def _to_ae_input(frames_2d):
    """(N, 9) → (N, 1, 9)  after StandardScaler transform."""
    return scaler_s2.transform(frames_2d).astype(np.float32) \
                    .reshape(-1, PAPER_AE_TIME_STEPS, PAPER_N_FEATURES)

X_ae_fit_normal = _to_ae_input(fit_frames_normal)
X_ae_val        = _to_ae_input(val_frames_all)
X_ae_test       = _to_ae_input(test_frames_all)
X_ae_unseen     = _to_ae_input(unseen_frames_all)
print('Stage-2 LSTM-AE training tensor :', X_ae_fit_normal.shape)

# Step 5: Train Stage 2 LSTM-Autoencoder
print('\n========== Training Stage 2 (Paper LSTM-Autoencoder) ==========')
ae = build_paper_lstm_autoencoder()
ae.summary()

# Paper specifies 100 epochs without early stopping. We add a generous-patience
# EarlyStopping (patience=20) purely to abort if the model has clearly plateaued.
cb_ae = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
]
ae.fit(
    X_ae_fit_normal, X_ae_fit_normal,
    validation_split=PAPER_AE_VAL_SPLIT,
    epochs=PAPER_AE_EPOCHS, batch_size=PAPER_AE_BATCH,
    callbacks=cb_ae, verbose=2,
)

# Step 6: Compute paper's threshold = mean + std of training reconstruction errors
print('\n========== Computing paper threshold (μ + σ of train recon errors) ==========')
recon_train = ae.predict(X_ae_fit_normal, batch_size=PAPER_AE_BATCH, verbose=0)
err_train   = compute_per_sample_mse(X_ae_fit_normal, recon_train)
mu_err      = float(err_train.mean())
sd_err      = float(err_train.std())
threshold   = mu_err + sd_err
print('Train recon error μ : {:.6f}'.format(mu_err))
print('Train recon error σ : {:.6f}'.format(sd_err))
print('Threshold (μ + σ)   : {:.6f}'.format(threshold))

# Quick sanity: how many training-normal frames would the threshold flag?
n_train_anomalous = int((err_train > threshold).sum())
print('Train-normal samples above threshold: {} / {}  ({:.2%})'
      .format(n_train_anomalous, len(err_train), n_train_anomalous / max(len(err_train), 1)))

# Step 7: Evaluate on held-out test set + zero_day (LOAO)
print('\n========== Evaluating on held-out test set ==========')

# Concatenate test + unseen windows for Stage 1 prediction
X_eval_seq = np.concatenate([X_test_s, X_unseen_s], axis=0)
y_eval_true_str = np.concatenate([
    le5.inverse_transform(y_test_enc),
    np.array(['zero_day'] * len(X_unseen_s), dtype=object),
])

# Stage 1 predictions
s1_probs = stage1.predict(X_eval_seq, batch_size=BATCH_SIZE, verbose=0)
s1_eval  = np.argmax(s1_probs, axis=1)

# Stage 2 inputs (paper frames + StandardScaler)
X_ae_eval = np.concatenate([X_ae_test, X_ae_unseen], axis=0)
recon_eval = ae.predict(X_ae_eval, batch_size=PAPER_AE_BATCH, verbose=0)
err_eval   = compute_per_sample_mse(X_ae_eval, recon_eval)

# Hybrid decision: Stage 1 fires → use Stage 1's label.
# Stage 1 says Normal → Stage 2 fires if err > threshold → 'zero_day' else 'Normal'
final_pred = []
for i in range(len(s1_eval)):
    if s1_eval[i] != normal_idx5:
        final_pred.append(le5.inverse_transform(np.array([s1_eval[i]]))[0])
    else:
        final_pred.append('zero_day' if err_eval[i] > threshold else 'Normal')

# Stage 2 binary view (Normal vs zero_day on the Stage1=Normal slice)
mask_s1_normal = s1_eval == normal_idx5
y_bin_true, y_bin_pred = [], []
for i in range(len(y_eval_true_str)):
    if not mask_s1_normal[i]:
        continue
    t = y_eval_true_str[i]
    if t not in ('Normal', 'zero_day'):
        continue
    y_bin_true.append(t)
    y_bin_pred.append('zero_day' if err_eval[i] > threshold else 'Normal')

labels_bin = ['Normal', 'zero_day']
if len(y_bin_true) > 0:
    print_metrics_block(
        'Stage 2 (Paper LSTM-AE + μ+σ threshold) — Normal vs zero_day',
        y_bin_true, y_bin_pred, labels_bin,
    )
    plot_confusion_heatmap(
        y_bin_true, y_bin_pred, labels_bin,
        'Stage 2 — new_model (Paper LSTM-AE)',
        'new_model_cm_stage2.png', figsize=(7, 6),
    )

# Final 5-class hybrid evaluation (Normal + 3 known + zero_day)
print_metrics_block(
    'Final hybrid — new_model (LOAO: {} unseen)'.format(UNSEEN_ATTACK),
    y_eval_true_str, final_pred, FINAL_LABELS,
)
plot_confusion_heatmap(
    y_eval_true_str, final_pred, FINAL_LABELS,
    'Final hybrid — new_model (Paper Stage 2, LOAO: {})'.format(UNSEEN_ATTACK),
    'new_model_cm_final.png', figsize=(11, 9),
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
print('  Threshold (μ + σ)  : {:.6f}'.format(threshold))

print('\n--- new_model.py complete (LOAO: {} unseen).  Paper Stage 2: LSTM-AE  ---'
      .format(UNSEEN_ATTACK))

# ── Download outputs from Colab ───────────────────────────────────────────────
if _IN_COLAB:
    from google.colab import files
    for fname in ['new_model_cm_stage1.png', 'new_model_cm_stage2.png', 'new_model_cm_final.png']:
        if os.path.isfile(fname):
            files.download(fname)
            print('Downloaded:', fname)
        else:
            print('Not found (skipped):', fname)
