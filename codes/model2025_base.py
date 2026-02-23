# ===============================
# Base Paper: Kim et al., "Adaptive Autoencoder-Based Intrusion Detection System
#             with Single Threshold for CAN Networks", Sensors 2025, 25, 4174.
# Dataset:   Car-Hacking Dataset (HCRL) — same as our solution (model2025_cursor.py)
#            https://ocslab.hksecurity.net/Datasets/car-hacking-dataset
#            Attributes: Timestamp, CAN ID (HEX), DLC, DATA[0-7], Flag (T=injected, R=normal)
#            Files: Attack-free (normal), DoS, Fuzzy, Spoofing gear, Spoofing RPM
# ===============================
# Pipeline (base paper): CAN ID only → 29-bit → N-frame windows → autoencoder (normal only)
#                       → single threshold via KDE → binary normal/attack
# ===============================

import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.integrate import quad
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Reshape
from tensorflow.keras.callbacks import EarlyStopping

# Optional: Colab. If running locally, set DATA_PATH to your Car-Hacking Dataset folder.
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DATA_PATH = os.environ.get('CAR_HACKING_DATA_PATH', '/content/drive/MyDrive/dataset/9) Car-Hacking Dataset')
except ImportError:
    DATA_PATH = os.environ.get('CAR_HACKING_DATA_PATH', os.path.join(os.path.dirname(__file__), 'car-hacking-dataset'))

data_path = DATA_PATH
print("Car-Hacking Dataset path:", data_path)
print("Pipeline: base paper (CAN ID only, N-frame autoencoder, KDE threshold) with Car-Hacking data.\n")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# Set to True to search N in [15, 64] and pick optimal by ERE (slow).
DO_FULL_N_SEARCH = False
N_DEFAULT = 40  # Paper's reported optimal N

# ===============================
# 1) Load Car-Hacking Dataset (same as our solution pipeline)
#    Source: https://ocslab.hksecurity.net/Datasets/car-hacking-dataset
#    Attributes: Timestamp, CAN ID (HEX), DLC, DATA[0-7], Flag (T=injected, R=normal)
# ===============================

# Column names per Car-Hacking Dataset 1.1
CAR_HACKING_COLUMNS = [
    'Timestamp', 'CAN_ID', 'DLC',
    'DATA0', 'DATA1', 'DATA2', 'DATA3', 'DATA4', 'DATA5', 'DATA6', 'DATA7',
    'Flag'
]

def parse_normal_txt(txt_path):
    """Parse attack-free (normal) log; returns sequence of CAN IDs (HEX in file)."""
    regex = r"Timestamp:\s*(\d+\.\d+)\s+ID:\s*(\w+)\s+000\s+DLC:\s*(\d+)\s+([\da-fA-F\s]+)"
    ids = []
    with open(txt_path, 'r') as f:
        for line in f:
            match = re.match(regex, line.strip())
            if match:
                can_id = int(match.group(2), 16)
                ids.append(can_id)
    return np.array(ids)

def is_attack_flag_car_hacking(flag_series):
    """Car-Hacking Dataset: T = injected (attack), R = normal (per dataset 1.1)."""
    s = flag_series.astype(str).str.strip().str.upper()
    if s.isin(['R', 'T']).any():
        return (s == 'T').values
    try:
        v = pd.to_numeric(flag_series, errors='coerce')
        if v.notna().all():
            return (v != 0).values
    except Exception:
        pass
    return np.ones(len(flag_series), dtype=bool)

def can_id_hex_to_int(series):
    """CAN ID in dataset is HEX (e.g. 043f). Convert to int."""
    out = []
    for v in series:
        if isinstance(v, (int, float)) and not np.isnan(v):
            out.append(int(v))
        elif isinstance(v, str):
            v = v.strip()
            if v.startswith(('0x', '0X')):
                out.append(int(v, 16))
            elif v.isdigit():
                out.append(int(v))
            else:
                out.append(int(v, 16))
        else:
            out.append(int(v))
    return np.array(out)

def load_attack_csv(csv_path):
    """Load one Car-Hacking attack CSV; return (CAN_IDs, is_attack per row)."""
    df = pd.read_csv(csv_path, header=None, names=CAR_HACKING_COLUMNS)
    can_ids = can_id_hex_to_int(df['CAN_ID'])
    is_attack = is_attack_flag_car_hacking(df['Flag'])
    return can_ids, is_attack

# Normal: attack-free dataset (same file layout as model2025_cursor.py)
normal_path = os.path.join(data_path, 'normal_run_data', 'normal_run_data.txt')
if not os.path.isfile(normal_path):
    normal_path = os.path.join(data_path, 'normal_run_data.txt')
normal_ids = parse_normal_txt(normal_path)

# Attack datasets (DoS, Fuzzy, Spoofing gear, Spoofing RPM)
dos_ids, dos_attack = load_attack_csv(os.path.join(data_path, 'dos_attack.csv'))
fuzzy_ids, fuzzy_attack = load_attack_csv(os.path.join(data_path, 'fuzzy_attack.csv'))
gear_ids, gear_attack = load_attack_csv(os.path.join(data_path, 'gear_spoofing.csv'))
rpm_ids, rpm_attack = load_attack_csv(os.path.join(data_path, 'rpm_spoofing.csv'))

attack_sequences = [
    (dos_ids, dos_attack, 'DoS'),
    (fuzzy_ids, fuzzy_attack, 'Fuzzy'),
    (gear_ids, gear_attack, 'Gear'),
    (rpm_ids, rpm_attack, 'RPM'),
]

# ===============================
# 2) CAN ID → 29-bit (zero-padding; paper Section 3.2)
# ===============================

def can_id_to_29bit(can_id):
    """11-bit CAN ID zero-padded to 29 bits (MSB first)."""
    id_11 = int(can_id) & 0x7FF
    bits = [0] * 18 + [((id_11 >> (10 - i)) & 1) for i in range(11)]
    return np.array(bits, dtype=np.float32)

def ids_to_windows(ids, is_attack=None, N=40, step=1):
    """
    ids: 1D array of CAN IDs (int).
    is_attack: optional 1D bool; if provided, label window as attack when any frame in window is attack.
    Returns: X (num_windows, N, 29), [optional] labels (num_windows,) bool = is_attack_window
    """
    vecs = np.array([can_id_to_29bit(i) for i in ids])
    num_windows = (len(vecs) - N) // step + 1
    if num_windows <= 0:
        return np.zeros((0, N, 29), dtype=np.float32), np.zeros(0, dtype=bool) if is_attack is not None else None
    X = np.zeros((num_windows, N, 29), dtype=np.float32)
    for i in range(num_windows):
        start = i * step
        X[i] = vecs[start : start + N]
    if is_attack is not None:
        labels = np.zeros(num_windows, dtype=bool)
        for i in range(num_windows):
            start = i * step
            labels[i] = np.any(is_attack[start : start + N])
        return X, labels
    return X, None

# ===============================
# 3) Build normal and attack windows; train / threshold / test split
# ===============================

def build_splits(N, step=1):
    # Normal: 80% train, 20% test (by windows)
    normal_X, _ = ids_to_windows(normal_ids, None, N=N, step=step)
    n_normal = len(normal_X)
    idx = np.arange(n_normal)
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE, shuffle=True)
    X_normal_train = normal_X[train_idx]
    X_normal_test = normal_X[test_idx]

    # Attack: 2/3 for threshold, 1/3 for test (per type)
    attack_windows_threshold = []  # list of (X, name)
    attack_windows_test = []
    for ids, is_att, name in attack_sequences:
        X_att, lab = ids_to_windows(ids, is_att, N=N, step=step)
        # only windows that contain at least one attack frame (paper: "Attack" = at least one)
        attack_only = lab
        X_att_only = X_att[attack_only]
        n_a = len(X_att_only)
        if n_a == 0:
            continue
        idx_a = np.arange(n_a)
        th_idx, te_idx = train_test_split(idx_a, test_size=1/3, random_state=RANDOM_STATE, shuffle=True)
        attack_windows_threshold.append((X_att_only[th_idx], name))
        attack_windows_test.append((X_att_only[te_idx], name))

    return X_normal_train, X_normal_test, attack_windows_threshold, attack_windows_test

# ===============================
# 4) Autoencoder (paper Section 3.3): Flatten → Dense(64, ReLU) → Dense(N*29, sigmoid) → Reshape
# ===============================

def build_autoencoder(N):
    # (N, 29) → N*29 → 64 → N*29 → (N, 29)
    flat_dim = N * 29
    inp = Input(shape=(N, 29))
    x = Flatten()(inp)
    x = Dense(64, activation='relu')(x)
    x = Dense(flat_dim, activation='sigmoid')(x)
    out = Reshape((N, 29))(x)
    model = Model(inp, out)
    model.compile(optimizer='adam', loss='mse')
    return model

# ===============================
# 5) KDE threshold (paper: intersection of KDE_normal and KDE_attack)
# ===============================

def find_threshold_kde(loss_normal, loss_attack, x_grid=None):
    """Threshold = argmin_x |KDE_normal(x) - KDE_attack(x)|."""
    if len(loss_normal) == 0 or len(loss_attack) == 0:
        return np.nan
    kde_n = stats.gaussian_kde(loss_normal)
    kde_a = stats.gaussian_kde(loss_attack)
    if x_grid is None:
        lo = min(loss_normal.min(), loss_attack.min())
        hi = max(loss_normal.max(), loss_attack.max())
        x_grid = np.linspace(lo, hi, 500)
    diff = np.abs(kde_n(x_grid) - kde_a(x_grid))
    idx = np.argmin(diff)
    return float(x_grid[idx])

# ===============================
# 6) ERE - Error Rate Estimation (paper Equation 8)
# ===============================

def ere_at_threshold(kde_normal, kde_attack, LTh):
    """ERE = P(normal above LTh) + P(attack below LTh)."""
    lo = float(min(kde_normal.dataset.min(), kde_attack.dataset.min()) - 1e-5)
    hi = float(max(kde_normal.dataset.max(), kde_attack.dataset.max()) + 1e-5)
    def pdf_n(x):
        return float(np.atleast_1d(kde_normal(x))[0])
    def pdf_a(x):
        return float(np.atleast_1d(kde_attack(x))[0])
    term1, _ = quad(pdf_n, LTh, hi, limit=100)
    term2, _ = quad(pdf_a, lo, LTh, limit=100)
    return term1 + term2

# ===============================
# 7) Train one model for given N; compute threshold and ERE
# ===============================

def train_and_threshold_for_N(N, X_normal_train, X_normal_test,
                              attack_windows_threshold, attack_windows_test,
                              epochs=30, batch_size=64, patience=5):
    model = build_autoencoder(N)
    model.fit(
        X_normal_train, X_normal_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)],
        verbose=0
    )

    # Losses on normal (training) and on each attack type (threshold portion)
    def mse_per_sample(X):
        pred = model.predict(X, verbose=0)
        return np.mean((X - pred) ** 2, axis=(1, 2))

    loss_normal = mse_per_sample(X_normal_train)
    thresholds = {}
    kdes_n = stats.gaussian_kde(loss_normal)
    for X_att, name in attack_windows_threshold:
        if len(X_att) == 0:
            continue
        loss_att = mse_per_sample(X_att)
        kdes_a = stats.gaussian_kde(loss_att)
        th = find_threshold_kde(loss_normal, loss_att)
        thresholds[name] = (th, kdes_n, kdes_a)

    # Same threshold for all? Use mean of thresholds; check max deviation
    th_vals = [v[0] for v in thresholds.values() if not np.isnan(v[0])]
    if len(th_vals) < 4:
        return None, np.inf, None, model

    th_mean = np.mean(th_vals)
    th_std = np.std(th_vals)
    # "Same" = within 2% of range or std very small
    if th_std > 1e-6 and th_std / (np.max(th_vals) - np.min(th_vals) + 1e-9) > 0.05:
        return None, np.inf, None, model

    # ERE at th_mean for each attack type, then average
    ere_sum = 0
    count = 0
    for name, (_, kdn, kda) in thresholds.items():
        ere_sum += ere_at_threshold(kdn, kda, th_mean)
        count += 1
    ere_avg = ere_sum / count if count else np.inf
    return th_mean, ere_avg, thresholds, model

# ===============================
# 8) Main: N selection (optional) or single N=40
# ===============================

if DO_FULL_N_SEARCH:
    N_range = range(15, 65)
    best_N = N_DEFAULT
    best_ere = np.inf
    best_threshold = None
    best_model = None
    best_splits = None

    for N in N_range:
        X_norm_tr, X_norm_te, att_th, att_te = build_splits(N, step=N)  # non-overlapping for speed
        if len(X_norm_tr) < 100:
            continue
        th, ere, _, model = train_and_threshold_for_N(
            N, X_norm_tr, X_norm_te, att_th, att_te,
            epochs=20, batch_size=128, patience=3
        )
        if th is not None and ere < best_ere:
            best_ere = ere
            best_N = N
            best_threshold = th
            best_model = model
            best_splits = (X_norm_tr, X_norm_te, att_th, att_te)
        print(f"N={N} threshold={th} ERE_avg={ere:.6f}")

    if best_model is None:
        print("No N gave same threshold for all attack types; using N=40.")
        best_N = N_DEFAULT
        X_norm_tr, X_norm_te, att_th, att_te = build_splits(best_N, step=1)
        _, _, _, best_model = train_and_threshold_for_N(best_N, X_norm_tr, X_norm_te, att_th, att_te)
        loss_n = np.mean((X_norm_tr - best_model.predict(X_norm_tr, verbose=0)) ** 2, axis=(1, 2))
        th_list = []
        for X_att, _ in att_th:
            if len(X_att) == 0:
                continue
            loss_a = np.mean((X_att - best_model.predict(X_att, verbose=0)) ** 2, axis=(1, 2))
            th_list.append(find_threshold_kde(loss_n, loss_a))
        best_threshold = np.mean(th_list) if th_list else (np.mean(loss_n) + 3 * np.std(loss_n))
        best_splits = (X_norm_tr, X_norm_te, att_th, att_te)

    N_opt = best_N
    threshold_opt = best_threshold
    model = best_model
    X_normal_train, X_normal_test, attack_windows_threshold, attack_windows_test = best_splits
else:
    N_opt = N_DEFAULT
    X_normal_train, X_normal_test, attack_windows_threshold, attack_windows_test = build_splits(N_opt, step=1)
    threshold_opt, _, _, model = train_and_threshold_for_N(
        N_opt, X_normal_train, X_normal_test,
        attack_windows_threshold, attack_windows_test
    )
    if threshold_opt is None:
        # Fallback: mean + 3*std on normal training loss
        pred_tr = model.predict(X_normal_train, verbose=0)
        loss_tr = np.mean((X_normal_train - pred_tr) ** 2, axis=(1, 2))
        threshold_opt = np.mean(loss_tr) + 3 * np.std(loss_tr)
        print("Using fallback threshold (mean + 3*std):", threshold_opt)

print("Optimal N:", N_opt)
print("Single threshold (KDE):", threshold_opt)

# ===============================
# 9) Evaluation on test set (paper Section 4)
# ===============================

def mse_per_sample(model, X):
    pred = model.predict(X, verbose=0)
    return np.mean((X - pred) ** 2, axis=(1, 2))

# Test: normal windows + attack windows (each type)
y_true = []
y_score = []  # MSE
# Normal test
loss_norm_te = mse_per_sample(model, X_normal_test)
y_true.extend([0] * len(loss_norm_te))  # 0 = normal
y_score.extend(loss_norm_te.tolist())
# Attack test
for X_att, _ in attack_windows_test:
    if len(X_att) == 0:
        continue
    loss_att = mse_per_sample(model, X_att)
    y_true.extend([1] * len(loss_att))  # 1 = attack
    y_score.extend(loss_att.tolist())

y_true = np.array(y_true)
y_score = np.array(y_score)
y_pred = (y_score > threshold_opt).astype(int)

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("\n--- Base paper model (CAN ID only, N={}, single threshold) ---".format(N_opt))
print("Accuracy:  {:.4f}".format(acc))
print("Precision: {:.4f}".format(prec))
print("Recall:    {:.4f}".format(rec))
print("F1-Score:  {:.4f}".format(f1))
print("Confusion matrix (true vs pred: [TN FP; FN TP]):")
print(confusion_matrix(y_true, y_pred))
