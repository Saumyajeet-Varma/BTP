# ===============================
# endsem_new_pipeline.py
# Two-stage CAN IDS aligned with CANGuard (arXiv:2603.25763v1) + anomaly second stage
#   Stage 1 — Conv1D stack -> stacked BiGRU -> additive attention -> MLP (multiclass)
#   Stage 2 — AAE trained on ground-truth Normal windows only; scores windows Stage 1 calls Normal
#   Hybrid — non-Normal Stage 1 -> known label; Stage1 Normal + score > T -> zero_day
# Dataset: Car-Hacking (16-D per-message features, sorted by time, sliding windows)
# ===============================

try:
    from google.colab import drive

    drive.mount("/content/drive", force_remount=False)
    _IN_COLAB = True
except ImportError:
    _IN_COLAB = False

import os
import re
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if _IN_COLAB:
    data_path = "/content/drive/MyDrive/dataset/9) Car-Hacking Dataset"
else:
    data_path = r"9) Car-Hacking Dataset"

USE_SUBSET = True
MAX_NORMAL = 50_000
MAX_PER_ATTACK_FILE = 50_000
SEQ_LEN = 12
BATCH_SIZE = 64
STAGE1_EPOCHS = 50
STAGE1_PATIENCE = 10
STAGE1_VAL_FRAC = 0.15
STAGE1_LR = 0.001
STAGE1_WEIGHT_DECAY = 0.001
STAGE2_NORMAL_PERCENTILE = 97.0
STAGE2_SCORE_AGG = "max"
STAGE2_SCORE_STATS = "train_normal"

AAE_EPOCHS = 160
AAE_BATCH = 512
DISC_STEPS = 2
INPUT_NOISE_STD = 0.01
LATENT_DIM = 24
LATENT_WEIGHT = 0.22

NUM_PROBE_FAKES = 4000
INCLUDE_UNIFORM_PROBE = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if weight is not None:
            self.register_buffer("ce_weight", weight)
        else:
            self.register_buffer("ce_weight", torch.empty(0))

    def forward(self, logits, targets):
        w = self.ce_weight if self.ce_weight.numel() > 0 else None
        ce = nn.functional.cross_entropy(
            logits, targets, weight=w, reduction="none", label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce)
        return ((1 - pt).pow(self.gamma) * ce).mean()


class CANGuardStage1(nn.Module):
    """
    CNN-GRU-Attention style stack (CANGuard paper).
    Input: (batch, n_features, seq_len) — multivariate CAN window.
    """

    def __init__(self, n_features, num_classes):
        super().__init__()
        d = 0.3
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            nn.Dropout(d),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),
            nn.Dropout(d),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(2),
            nn.Dropout(d),
        )
        gru_in = 256
        h1, h2 = 128, 64
        self.gru1 = nn.GRU(
            gru_in,
            h1,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.gru2 = nn.GRU(h1 * 2, h2, num_layers=1, batch_first=True, bidirectional=True, dropout=0.0)
        attn_dim = h2 * 2
        self.attn_score = nn.Linear(attn_dim, 1, bias=True)
        self.drop_post = nn.Dropout(d)
        self.fc1 = nn.Linear(attn_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, num_classes)

    def forward(self, x):
        h = self.cnn(x)
        h = h.permute(0, 2, 1)
        h, _ = self.gru1(h)
        h = nn.functional.dropout(h, 0.3, self.training)
        h, _ = self.gru2(h)
        u = torch.tanh(h)
        e = self.attn_score(u)
        alpha = torch.softmax(e, dim=1)
        c = (alpha * h).sum(dim=1)
        z = self.drop_post(c)
        z = nn.functional.relu(self.fc1(z))
        z = nn.functional.dropout(z, 0.3, self.training)
        z = nn.functional.relu(self.fc2(z))
        z = nn.functional.dropout(z, 0.3, self.training)
        return self.out(z)


class Encoder(nn.Module):
    def __init__(self, n_in, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.05),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.05),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, n_out),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class LatentDiscriminator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(latent_dim, 64)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.utils.spectral_norm(nn.Linear(64, 32)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.utils.spectral_norm(nn.Linear(32, 1)),
        )

    def forward(self, z):
        return self.net(z)


def parse_line(line):
    regex = r"Timestamp:\s*(\d+\.\d+)\s+ID:\s*(\w+)\s+000\s+DLC:\s*(\d+)\s+([\da-fA-F\s]+)"
    match = re.match(regex, line.strip())
    if match:
        timestamp = float(match.group(1))
        can_id = int(match.group(2), 16)
        dlc = int(match.group(3))
        data = [int(byte, 16) for byte in match.group(4).split()]
        data = (data + [0] * 8)[:8]
        return {"Timestamp": timestamp, "CAN_ID": can_id, "DLC": dlc, "DATA": data}
    return None


def normal_txt_path(base):
    p1 = os.path.join(base, "normal_run_data.txt")
    p2 = os.path.join(base, "normal_run_data", "normal_run_data.txt")
    if os.path.isfile(p1):
        return p1
    if os.path.isfile(p2):
        return p2
    return p1


def load_full_dataframe(base_path, use_subset, max_normal, max_per_attack):
    file_path = normal_txt_path(base_path)
    rows = []
    with open(file_path, "r") as f:
        for line in f:
            if use_subset and len(rows) >= max_normal:
                break
            p = parse_line(line)
            if p:
                rows.append(p)

    df_normal = pd.DataFrame(rows)
    for i in range(8):
        df_normal["DATA{}".format(i)] = df_normal["DATA"].apply(lambda x, i=i: x[i] if i < len(x) else 0)
    df_normal.drop(columns=["DATA"], inplace=True)
    df_normal["Label"] = "Normal"

    column_names = ["Timestamp", "CAN_ID", "DLC"] + ["DATA{}".format(i) for i in range(8)] + ["Flag"]

    def hex_to_int(s):
        s = str(s).strip()
        if re.match(r"^[0-9a-fA-F]+$", s):
            return int(s, 16)
        return 0

    def convert_numeric_columns(df, cols):
        for col in cols:
            if col == "CAN_ID" or col.startswith("DATA"):
                df[col] = df[col].map(hex_to_int).astype(int)
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        return df

    def label_from_flag(flag_series, attack_name):
        return [
            "Normal" if v == "R" else attack_name if v == "T" else "Normal"
            for v in flag_series.astype(str).str.strip().str.upper()
        ]

    cols_to_process = ["CAN_ID", "DLC"] + ["DATA{}".format(i) for i in range(8)]
    nrows = max_per_attack if use_subset else None
    attack_files = [
        ("dos_attack.csv", "DoS"),
        ("fuzzy_attack.csv", "Fuzzy"),
        ("gear_spoofing.csv", "Gear"),
        ("rpm_spoofing.csv", "RPM"),
    ]
    parts = [df_normal]
    for fname, aname in attack_files:
        df = pd.read_csv(os.path.join(base_path, fname), header=None, names=column_names, nrows=nrows)
        df = convert_numeric_columns(df, cols_to_process)
        df["Label"] = label_from_flag(df["Flag"], aname)
        parts.append(df)

    full_df = pd.concat(parts, ignore_index=True)
    if "Flag" in full_df.columns:
        full_df.drop(columns=["Flag"], inplace=True)

    base_features = ["CAN_ID", "DLC"] + ["DATA{}".format(i) for i in range(8)]
    full_df = full_df.sort_values("Timestamp").reset_index(drop=True)
    full_df["IAT"] = full_df["Timestamp"].diff().fillna(0).clip(upper=1.0)
    freq = full_df["CAN_ID"].value_counts(normalize=True)
    full_df["CAN_ID_freq"] = full_df["CAN_ID"].map(freq)
    data_cols = ["DATA{}".format(i) for i in range(8)]

    def row_entropy(row):
        vals = row.values.astype(float)
        vals = vals[vals > 0]
        if len(vals) == 0:
            return 0.0
        p = vals / vals.sum()
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

    full_df["byte_entropy"] = full_df[data_cols].apply(row_entropy, axis=1)
    full_df["byte_sum"] = full_df[data_cols].sum(axis=1)
    full_df["byte_range"] = full_df[data_cols].max(axis=1) - full_df[data_cols].min(axis=1)
    full_df["byte_std"] = full_df[data_cols].std(axis=1)
    features = base_features + ["IAT", "CAN_ID_freq", "byte_entropy", "byte_sum", "byte_range", "byte_std"]
    return full_df, features


def make_sliding_windows(X, y, seq_len):
    """X: (N, F), y: (N,) -> Xw: (N-seq_len+1, F, seq_len), yw labels at last timestep."""
    if len(X) < seq_len:
        raise ValueError("Need len(X) >= seq_len")
    n = len(X) - seq_len + 1
    f = X.shape[1]
    Xw = np.zeros((n, f, seq_len), dtype=np.float32)
    yw = np.zeros(n, dtype=np.int64)
    for i in range(n):
        sl = X[i : i + seq_len]
        Xw[i] = sl.T
        yw[i] = y[i + seq_len - 1]
    return Xw, yw


def predict_stage1_labels(model, loader):
    model.eval()
    out = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            out.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(out)


def compute_recon_lat_stats(enc, dec, X_np):
    enc.eval()
    dec.eval()
    bs = 2048
    recons, latns = [], []
    for i in range(0, len(X_np), bs):
        chunk = torch.FloatTensor(X_np[i : i + bs]).to(device)
        with torch.no_grad():
            z = enc(chunk)
            r = dec(z)
        recons.append(((chunk - r) ** 2).mean(dim=1).cpu().numpy())
        latns.append(torch.norm(z, dim=1).cpu().numpy())
    recon_err = np.concatenate(recons)
    lat_norm = np.concatenate(latns)
    return (
        float(recon_err.mean()),
        float(recon_err.std() + 1e-8),
        float(lat_norm.mean()),
        float(lat_norm.std() + 1e-8),
    )


def combined_scores(enc, dec, X_np, recon_mean, recon_std, lat_mean, lat_std):
    enc.eval()
    dec.eval()
    bs = 4096
    scores = []
    for i in range(0, len(X_np), bs):
        x = torch.FloatTensor(X_np[i : i + bs]).to(device)
        with torch.no_grad():
            z = enc(x)
            recon = dec(z)
            recon_err = ((x - recon) ** 2).mean(dim=1).cpu().numpy()
            lat_norm = torch.norm(z, dim=1).cpu().numpy()
        rs = np.abs(recon_err - recon_mean) / recon_std
        ls = np.abs(lat_norm - lat_mean) / lat_std
        if STAGE2_SCORE_AGG == "max":
            scores.append(np.maximum(rs, ls))
        else:
            scores.append((1.0 - LATENT_WEIGHT) * rs + LATENT_WEIGHT * ls)
    return np.concatenate(scores)


def train_stage1(model, train_loader, val_loader, class_weights, epochs, patience):
    crit = FocalLoss(weight=class_weights, gamma=2.0, label_smoothing=0.05)
    opt = optim.Adam(model.parameters(), lr=STAGE1_LR, weight_decay=STAGE1_WEIGHT_DECAY)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    best_vl = float("inf")
    best_state = None
    bad = 0
    hist_tr, hist_val = [], []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tr_loss, tr_n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * xb.size(0)
            tr_n += xb.size(0)
        tr_loss /= max(tr_n, 1)
        model.eval()
        vl_loss, vl_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                vl_loss += crit(model(xb), yb).item() * xb.size(0)
                vl_n += xb.size(0)
        vl_loss /= max(vl_n, 1)
        hist_tr.append(tr_loss)
        hist_val.append(vl_loss)
        sched.step(vl_loss)
        if vl_loss < best_vl:
            best_vl = vl_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    return time.time() - t0, hist_tr, hist_val


def train_stage2_aae(enc, dec, disc, X_normal_np, X_norm_val_t, ae_loss_log):
    opt_ae = optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=8e-4, weight_decay=1e-4)
    opt_disc = optim.AdamW(disc.parameters(), lr=2e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    opt_gen = optim.AdamW(enc.parameters(), lr=2e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    sched_ae = optim.lr_scheduler.ReduceLROnPlateau(opt_ae, mode="min", factor=0.5, patience=8, min_lr=1e-5)
    recon_loss_fn = nn.SmoothL1Loss(beta=0.05)
    adv_loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_normal_np)),
        batch_size=AAE_BATCH,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
    )
    best_mse = float("inf")
    best_pack = (None, None, None)
    pat = 0
    t0 = time.time()
    for _epoch in range(AAE_EPOCHS):
        enc.train()
        dec.train()
        disc.train()
        ep_ae = 0.0
        nb = 0
        for (real_cpu,) in loader:
            real = real_cpu.to(device, non_blocking=True)
            noisy = (real + INPUT_NOISE_STD * torch.randn_like(real)).clamp(0.0, 1.0)
            bs = real.size(0)
            z = enc(noisy)
            recon = dec(z)
            ae_loss = recon_loss_fn(recon, real)
            opt_ae.zero_grad()
            ae_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 1.0)
            opt_ae.step()
            for _ in range(DISC_STEPS):
                z_real = torch.randn(bs, LATENT_DIM, device=device)
                z_fake = enc(real).detach()
                d_loss = 0.5 * (
                    adv_loss_fn(disc(z_real), torch.full((bs, 1), 0.9, device=device))
                    + adv_loss_fn(disc(z_fake), torch.full((bs, 1), 0.1, device=device))
                )
                opt_disc.zero_grad()
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
                opt_disc.step()
            z_adv = enc(real)
            g_loss = adv_loss_fn(disc(z_adv), torch.full((bs, 1), 0.9, device=device))
            opt_gen.zero_grad()
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            opt_gen.step()
            ep_ae += ae_loss.item()
            nb += 1
        ae_loss_log.append(ep_ae / max(nb, 1))
        enc.eval()
        dec.eval()
        with torch.no_grad():
            vz = enc(X_norm_val_t)
            vr = dec(vz)
            val_mse = ((X_norm_val_t - vr) ** 2).mean(dim=1).mean().item()
        sched_ae.step(val_mse)
        if val_mse < best_mse:
            best_mse = val_mse
            best_pack = (
                {k: v.detach().clone() for k, v in enc.state_dict().items()},
                {k: v.detach().clone() for k, v in dec.state_dict().items()},
                {k: v.detach().clone() for k, v in disc.state_dict().items()},
            )
            pat = 0
        else:
            pat += 1
        if pat >= 18:
            break
    if best_pack[0] is not None:
        enc.load_state_dict(best_pack[0])
        dec.load_state_dict(best_pack[1])
        disc.load_state_dict(best_pack[2])
    return time.time() - t0, best_mse


def flatten_windows(Xw):
    """(N, F, T) -> (N, F*T) for AE."""
    n = Xw.shape[0]
    return Xw.reshape(n, -1)


if not os.path.isdir(data_path):
    print("ERROR: Dataset not found at:", repr(data_path))
    print("Set data_path to your Car-Hacking folder (contains dos_attack.csv, etc.).")
else:
    full_df, features = load_full_dataframe(data_path, USE_SUBSET, MAX_NORMAL, MAX_PER_ATTACK_FILE)
    X = full_df[features].values.astype(np.float64)
    y_str = full_df["Label"].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y_str)
    num_classes = len(le.classes_)
    normal_idx = int(le.transform(["Normal"])[0])
    n_feat = len(features)

    X_trf, X_test, y_trf, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc
    )
    scaler = MinMaxScaler()
    X_trf_s = scaler.fit_transform(X_trf)
    X_test_s = scaler.transform(X_test)

    X_train, X_cal, y_train, y_cal = train_test_split(
        X_trf_s, y_trf, test_size=STAGE1_VAL_FRAC, random_state=RANDOM_STATE, stratify=y_trf
    )

    X_train_w, y_train_w = make_sliding_windows(X_train, y_train, SEQ_LEN)
    X_cal_w, y_cal_w = make_sliding_windows(X_cal, y_cal, SEQ_LEN)
    X_test_w, y_test_w = make_sliding_windows(X_test_s, y_test, SEQ_LEN)

    flat_dim = n_feat * SEQ_LEN

    X_train_t = torch.FloatTensor(X_train_w)
    y_train_t = torch.LongTensor(y_train_w)
    X_cal_t = torch.FloatTensor(X_cal_w)
    y_cal_t = torch.LongTensor(y_cal_w)
    X_test_t = torch.FloatTensor(X_test_w)
    y_test_t = torch.LongTensor(y_test_w)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
    )
    cal_loader = DataLoader(TensorDataset(X_cal_t, y_cal_t), batch_size=BATCH_SIZE * 2, shuffle=False, pin_memory=True)
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE * 2, shuffle=False, pin_memory=True
    )

    cw = compute_class_weight("balanced", classes=np.unique(y_train_w), y=y_train_w)
    class_weights = torch.FloatTensor(cw).to(device)

    stage1 = CANGuardStage1(n_feat, num_classes).to(device)
    print("\n--- Stage 1: CANGuard-style (Conv1D -> BiGRU -> attention) | SEQ_LEN={} ---".format(SEQ_LEN))
    t1, h1_tr, h1_val = train_stage1(
        stage1, train_loader, cal_loader, class_weights, STAGE1_EPOCHS, STAGE1_PATIENCE
    )
    print("Stage 1 time (s):", round(t1, 1))

    y_cal_pred = predict_stage1_labels(stage1, cal_loader)
    y_test_pred = predict_stage1_labels(stage1, test_loader)

    acc1 = accuracy_score(y_test_w, y_test_pred)
    p1w, r1w, _, _ = precision_recall_fscore_support(
        y_test_w, y_test_pred, average="weighted", zero_division=0
    )
    p1m, r1m, _, _ = precision_recall_fscore_support(
        y_test_w, y_test_pred, average="macro", zero_division=0
    )
    f1w = f1_score(y_test_w, y_test_pred, average="weighted", zero_division=0)
    f1m = f1_score(y_test_w, y_test_pred, average="macro", zero_division=0)
    cm1 = confusion_matrix(y_test_w, y_test_pred, labels=np.arange(num_classes))
    print("\n" + "=" * 70)
    print("STAGE 1 — Test metrics (multiclass, window labels)")
    print("=" * 70)
    print("accuracy_score:  {:.4f}".format(acc1))
    print(
        "precision (weighted): {:.4f} | recall (weighted): {:.4f} | f1_score (weighted): {:.4f}".format(
            p1w, r1w, f1w
        )
    )
    print(
        "precision (macro):    {:.4f} | recall (macro):    {:.4f} | f1_score (macro):    {:.4f}".format(
            p1m, r1m, f1m
        )
    )
    print("confusion_matrix [rows=true, cols=pred] | order:", list(le.classes_))
    print(cm1)
    print("\nPer-class classification_report:")
    print(classification_report(y_test_w, y_test_pred, target_names=le.classes_, zero_division=0))

    Xn_flat_train = flatten_windows(X_train_w[y_train_w == normal_idx])
    if len(Xn_flat_train) < 100:
        print("ERROR: Too few normal windows for Stage 2.")
    else:
        Xn_tr, Xn_val = train_test_split(Xn_flat_train, test_size=0.12, random_state=RANDOM_STATE)
        X_norm_val_t = torch.FloatTensor(Xn_val).to(device)

        enc = Encoder(flat_dim, LATENT_DIM).to(device)
        dec = Decoder(LATENT_DIM, flat_dim).to(device)
        disc = LatentDiscriminator(LATENT_DIM).to(device)
        ae_curve = []
        print("\n--- Stage 2: AAE on flattened Normal windows only (dim={}) ---".format(flat_dim))
        t2, best_mse = train_stage2_aae(enc, dec, disc, Xn_tr, X_norm_val_t, ae_curve)
        print("Stage 2 time (s):", round(t2, 1), "| best val MSE:", round(best_mse, 6))

        if STAGE2_SCORE_STATS == "train_normal":
            recon_mean, recon_std, lat_mean, lat_std = compute_recon_lat_stats(enc, dec, Xn_tr)
            stats_note = "stats from AAE train normals"
        else:
            cal_normal_mask = y_cal_w == normal_idx
            passed_s1 = y_cal_pred == normal_idx
            pm = cal_normal_mask & passed_s1
            if pm.sum() < 50:
                pm = cal_normal_mask
            recon_mean, recon_std, lat_mean, lat_std = compute_recon_lat_stats(enc, dec, flatten_windows(X_cal_w[pm]))
            stats_note = "stats from cal pipeline-normal windows"

        cal_normal_mask = y_cal_w == normal_idx
        passed_s1 = y_cal_pred == normal_idx
        pipeline_normal_mask = cal_normal_mask & passed_s1
        if pipeline_normal_mask.sum() < 50:
            pipeline_normal_mask = cal_normal_mask

        val_scores = combined_scores(
            enc,
            dec,
            flatten_windows(X_cal_w[pipeline_normal_mask]),
            recon_mean,
            recon_std,
            lat_mean,
            lat_std,
        )
        threshold = float(np.percentile(val_scores, STAGE2_NORMAL_PERCENTILE))
        print("\nStage 2 | {} | agg={} | threshold ({} pct cal S1-pass normals): {:.6f}".format(
            stats_note, STAGE2_SCORE_AGG, STAGE2_NORMAL_PERCENTILE, threshold
        ))

        X_test_flat = flatten_windows(X_test_w)
        test_scores = combined_scores(enc, dec, X_test_flat, recon_mean, recon_std, lat_mean, lat_std)

        pred_names = le.inverse_transform(y_test_pred)
        normal_str = le.classes_[normal_idx]
        final_label = []
        for i in range(len(y_test_w)):
            if y_test_pred[i] != normal_idx:
                final_label.append(pred_names[i])
            elif test_scores[i] > threshold:
                final_label.append("zero_day")
            else:
                final_label.append(normal_str)
        final_label = np.array(final_label)

        true_names = le.inverse_transform(y_test_w)
        hyb_labels = sorted(set(true_names.tolist()) | set(final_label.tolist()))
        acc_h = accuracy_score(true_names, final_label)
        ph_w, rh_w, _, _ = precision_recall_fscore_support(
            true_names, final_label, labels=hyb_labels, average="weighted", zero_division=0
        )
        ph_m, rh_m, _, _ = precision_recall_fscore_support(
            true_names, final_label, labels=hyb_labels, average="macro", zero_division=0
        )
        fh_w = f1_score(true_names, final_label, labels=hyb_labels, average="weighted", zero_division=0)
        fh_m = f1_score(true_names, final_label, labels=hyb_labels, average="macro", zero_division=0)
        cm_h = confusion_matrix(true_names, final_label, labels=hyb_labels)

        true_is_attack = y_test_w != normal_idx
        hybrid_is_attack = final_label != normal_str
        hb_acc = accuracy_score(true_is_attack, hybrid_is_attack)
        hb_p, hb_r, _, _ = precision_recall_fscore_support(
            true_is_attack, hybrid_is_attack, average="binary", pos_label=1, zero_division=0
        )
        hb_f1 = f1_score(true_is_attack, hybrid_is_attack, average="binary", pos_label=1, zero_division=0)
        cm_h_bin = confusion_matrix(
            true_is_attack.astype(int), hybrid_is_attack.astype(int), labels=[0, 1]
        )

        s2_mask = y_test_pred == normal_idx
        n_s2 = int(s2_mask.sum())
        print("\n" + "=" * 70)
        print("STAGE 2 — zero_day eligibility (Car-Hacking has no true unknown class)")
        print("=" * 70)
        n_norm_on_s2 = int(np.sum((y_test_w == normal_idx) & s2_mask))
        n_true_atk_on_s2 = int(np.sum((y_test_w != normal_idx) & s2_mask))
        print(
            "Test windows with Stage1=Normal: {} | true Normal: {} | true attack (FN): {}".format(
                n_s2, n_norm_on_s2, n_true_atk_on_s2
            )
        )
        if n_true_atk_on_s2 > 0:
            atk_scores = test_scores[(y_test_w != normal_idx) & s2_mask]
            print(
                "  FN attacks fraction score>T: {:.1%}".format(float(np.mean(atk_scores > threshold)))
            )
        print("  Hybrid rows labeled zero_day:", int(np.sum(final_label == "zero_day")))
        print(
            "  Cal S1-pass normal FPR est.: {:.2%}".format(float(np.mean(val_scores > threshold)))
        )

        if n_s2 > 0:
            y_s2_true = (y_test_w[s2_mask] != normal_idx).astype(int)
            y_s2_pred = (test_scores[s2_mask] > threshold).astype(int)
            acc_s2 = accuracy_score(y_s2_true, y_s2_pred)
            p2, r2, _, _ = precision_recall_fscore_support(
                y_s2_true, y_s2_pred, average="binary", pos_label=1, zero_division=0
            )
            f2 = f1_score(y_s2_true, y_s2_pred, average="binary", pos_label=1, zero_division=0)
            cm_s2 = confusion_matrix(y_s2_true, y_s2_pred, labels=[0, 1])
            print("\n" + "=" * 70)
            print("STAGE 2 — Test metrics (binary anomaly, gated: Stage1 predicted Normal only)")
            print("=" * 70)
            print("accuracy_score:  {:.4f}".format(acc_s2))
            print(
                "precision (binary, pos=anomaly): {:.4f} | recall: {:.4f} | f1_score: {:.4f}".format(
                    p2, r2, f2
                )
            )
            print(
                "confusion_matrix [rows=true, cols=pred] | labels [0=normal, 1=attack] "
                "| pred 0=pass(score<=T), 1=anomaly(score>T)"
            )
            print(cm_s2)
            print("\nPer-class classification_report:")
            print(
                classification_report(
                    y_s2_true,
                    y_s2_pred,
                    labels=[0, 1],
                    target_names=["true_normal", "true_attack"],
                    zero_division=0,
                )
            )
        else:
            acc_s2 = float("nan")
            p2 = r2 = f2 = float("nan")
            cm_s2 = None
            print("\nSTAGE 2 — gated path: no test windows with Stage1=Normal; skip Stage 2 metrics.")

        y_s2_all_true = (y_test_w != normal_idx).astype(int)
        y_s2_all_pred = (test_scores > threshold).astype(int)
        acc_s2_all = accuracy_score(y_s2_all_true, y_s2_all_pred)
        p2a, r2a, _, _ = precision_recall_fscore_support(
            y_s2_all_true, y_s2_all_pred, average="binary", pos_label=1, zero_division=0
        )
        f2a = f1_score(y_s2_all_true, y_s2_all_pred, average="binary", pos_label=1, zero_division=0)
        cm_s2_all = confusion_matrix(y_s2_all_true, y_s2_all_pred, labels=[0, 1])
        print("\n" + "=" * 70)
        print("STAGE 2 — Test metrics (binary, diagnostic: score>T on ALL test windows)")
        print("(Ignores Stage 1 gate; shows raw anomaly detector vs ground-truth attack)")
        print("=" * 70)
        print("accuracy_score:  {:.4f}".format(acc_s2_all))
        print(
            "precision (binary, pos=attack): {:.4f} | recall: {:.4f} | f1_score: {:.4f}".format(
                p2a, r2a, f2a
            )
        )
        print(
            "confusion_matrix [rows=true, cols=pred] | [0=normal, 1=attack] | pred from score>T only"
        )
        print(cm_s2_all)
        try:
            print(
                "ROC-AUC (scores, attack vs normal on all test windows):",
                round(roc_auc_score(true_is_attack.astype(int), test_scores), 4),
            )
        except Exception:
            pass

        print("\n" + "=" * 70)
        print("COMBINED / HYBRID — Test metrics (multiclass: known + Normal + zero_day)")
        print("=" * 70)
        print("accuracy_score:  {:.4f}".format(acc_h))
        print(
            "precision (weighted): {:.4f} | recall (weighted): {:.4f} | f1_score (weighted): {:.4f}".format(
                ph_w, rh_w, fh_w
            )
        )
        print(
            "precision (macro):    {:.4f} | recall (macro):    {:.4f} | f1_score (macro):    {:.4f}".format(
                ph_m, rh_m, fh_m
            )
        )
        print("confusion_matrix [rows=true, cols=pred] | label order:", hyb_labels)
        print(cm_h)
        print("\nPer-class classification_report:")
        print(classification_report(true_names, final_label, labels=hyb_labels, zero_division=0))

        print("\n" + "=" * 70)
        print("COMBINED / HYBRID — Test metrics (binary: any attack vs final Normal)")
        print("Positive = predicted attack path (known label or zero_day)")
        print("=" * 70)
        print("accuracy_score:  {:.4f}".format(hb_acc))
        print(
            "precision (binary, pos=attack): {:.4f} | recall: {:.4f} | f1_score: {:.4f}".format(
                hb_p, hb_r, hb_f1
            )
        )
        print(
            "confusion_matrix [rows=true, cols=pred] | [0=benign, 1=attack] "
            "| pred 0=final Normal, 1=any attack / zero_day"
        )
        print(cm_h_bin)

        print("\n" + "=" * 70)
        print("METRICS SUMMARY (test set)")
        print("=" * 70)
        print(
            "Stage1 multiclass | acc {:.4f} | P_w {:.4f} R_w {:.4f} F1_w {:.4f} | P_m {:.4f} R_m {:.4f} F1_m {:.4f}".format(
                acc1, p1w, r1w, f1w, p1m, r1m, f1m
            )
        )
        if n_s2 > 0:
            print(
                "Stage2 gated      | acc {:.4f} | P {:.4f} R {:.4f} F1 {:.4f}".format(acc_s2, p2, r2, f2)
            )
        else:
            print("Stage2 gated      | (no Stage1=Normal windows on test)")
        print(
            "Stage2 all-score  | acc {:.4f} | P {:.4f} R {:.4f} F1 {:.4f} (score>T vs true attack)".format(
                acc_s2_all, p2a, r2a, f2a
            )
        )
        print(
            "Hybrid multiclass | acc {:.4f} | P_w {:.4f} R_w {:.4f} F1_w {:.4f} | P_m {:.4f} R_m {:.4f} F1_m {:.4f}".format(
                acc_h, ph_w, rh_w, fh_w, ph_m, rh_m, fh_m
            )
        )
        print(
            "Hybrid binary     | acc {:.4f} | P {:.4f} R {:.4f} F1 {:.4f}".format(hb_acc, hb_p, hb_r, hb_f1)
        )
        rng = np.random.default_rng(RANDOM_STATE)
        uni = rng.random((NUM_PROBE_FAKES, flat_dim)).astype(np.float32)
        fake_scores_uni = combined_scores(enc, dec, uni, recon_mean, recon_std, lat_mean, lat_std)
        print("Uniform[0,1]^d fakes: frac score>T = {:.1%}".format(float(np.mean(fake_scores_uni > threshold))))
        if INCLUDE_UNIFORM_PROBE:
            ver_true = np.array([0] * (pipeline_normal_mask.sum()) + [1] * NUM_PROBE_FAKES)
            ver_scores = np.concatenate(
                [
                    combined_scores(
                        enc,
                        dec,
                        flatten_windows(X_cal_w[pipeline_normal_mask]),
                        recon_mean,
                        recon_std,
                        lat_mean,
                        lat_std,
                    ),
                    fake_scores_uni,
                ]
            )
            ver_pred = (ver_scores > threshold).astype(int)
            print(
                "Balanced probe PR (0=cal normal, 1=uniform fake):",
                classification_report(ver_true, ver_pred, labels=[0, 1], target_names=["cal_norm", "uniform"], zero_division=0),
            )

        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, len(h1_tr) + 1), h1_tr, label="train")
        ax.plot(range(1, len(h1_val) + 1), h1_val, label="val")
        ax.set_title("Stage 1 loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
        ax.set_title("Stage 1 confusion")
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, len(ae_curve) + 1), ae_curve)
        ax.set_title("Stage 2 AE recon term (epoch mean)")
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(test_scores[y_test_w == normal_idx], bins=40, alpha=0.5, label="true normal", density=True)
        ax.hist(test_scores[y_test_w != normal_idx], bins=40, alpha=0.5, label="true attack", density=True)
        ax.axvline(threshold, color="k", linestyle="--", label="T")
        ax.set_title("Stage 2 scores (all test windows)")
        ax.legend()
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(hyb_labels)), 6))
        sns.heatmap(cm_h, annot=True, fmt="d", cmap="Purples", xticklabels=hyb_labels, yticklabels=hyb_labels, ax=ax)
        ax.set_title("Hybrid confusion")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.show()

    print("\nDone: endsem_new_pipeline (CANGuard S1 + AAE S2 + hybrid).")
