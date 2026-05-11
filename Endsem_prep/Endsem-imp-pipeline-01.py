# ===============================
# Endsem-imp-pipeline-01
# Two-stage CAN intrusion detection (PyTorch)
#   Stage 1 — Multi-class (known attacks + Normal), CANForge-style CNN-BiLSTM
#   Stage 2 — AAE on traffic Stage 1 calls Normal; threshold from true normals on that path
# Dataset: Car-Hacking (same feature pipeline as CANForge notebook)
# Style aligned with: codes/model2025_stage2_cursor.py
# ===============================

# ===============================
# Optional: Google Colab
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
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ===============================
# Reproducibility
# ===============================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# ===============================
# Dataset Path & Tunables
# ===============================
if _IN_COLAB:
    data_path = "/content/drive/MyDrive/dataset/9) Car-Hacking Dataset"
else:
    data_path = r"9) Car-Hacking Dataset"

USE_SUBSET = True
MAX_NORMAL = 50_000
MAX_PER_ATTACK_FILE = 50_000
BATCH_SIZE = 256
STAGE1_EPOCHS = 50
STAGE1_PATIENCE = 8
STAGE1_VAL_FRAC = 0.15
STAGE2_NORMAL_PERCENTILE = 99.5

AAE_EPOCHS = 180
AAE_BATCH = 1024
DISC_STEPS = 2
INPUT_NOISE_STD = 0.01
LATENT_DIM = 16
LATENT_WEIGHT = 0.22

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ===============================
# Stage 1: PyTorch — Focal loss + CANForge-style backbone
# ===============================


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


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        se = self.fc(x.mean(dim=2)).unsqueeze(2)
        return x * se


class CANForgeStage1(nn.Module):
    def __init__(self, n_features, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(1, 32, 1, padding=0), nn.ReLU(), nn.BatchNorm1d(32))
        self.conv3 = nn.Sequential(nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(32))
        self.conv5 = nn.Sequential(nn.Conv1d(1, 32, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(32))
        self.drop1 = nn.Dropout(0.28)
        self.se = SEBlock(96, reduction=4)
        self.conv_res = nn.Sequential(nn.Conv1d(96, 96, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(96))
        self.drop2 = nn.Dropout(0.28)
        self.lstm1 = nn.LSTM(96, 64, batch_first=True, bidirectional=True, dropout=0.28)
        self.bn_lstm1 = nn.BatchNorm1d(128)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True, bidirectional=True, dropout=0.28)
        self.bn_lstm2 = nn.BatchNorm1d(128)
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.38),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.28),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        multi = torch.cat([self.conv1(x), self.conv3(x), self.conv5(x)], dim=1)
        multi = self.drop1(self.se(multi))
        res = torch.relu(self.conv_res(multi) + multi)
        res = self.drop2(res)
        lstm_in = res.permute(0, 2, 1)
        o1, _ = self.lstm1(lstm_in)
        o1 = self.bn_lstm1(o1.permute(0, 2, 1)).permute(0, 2, 1)
        o2, _ = self.lstm2(o1)
        o2 = self.bn_lstm2(o2.permute(0, 2, 1)).permute(0, 2, 1)
        return self.classifier((o2 + o1).mean(dim=1))


class Encoder(nn.Module):
    def __init__(self, n_features, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.05),
            nn.Linear(64, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.05),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, n_features),
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


# ===============================
# Parse Normal TXT File
# ===============================
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


# ===============================
# Load Attack CSVs + Feature Engineering
# ===============================
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


def predict_stage1_labels(model, loader):
    model.eval()
    out = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            out.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(out)


def combined_scores(enc, dec, X_np, recon_mean, recon_std, lat_mean, lat_std):
    enc.eval()
    dec.eval()
    x = torch.FloatTensor(X_np).to(device)
    with torch.no_grad():
        z = enc(x)
        recon = dec(z)
        recon_err = ((x - recon) ** 2).mean(dim=1).cpu().numpy()
        lat_norm = torch.norm(z, dim=1).cpu().numpy()
    rs = np.abs(recon_err - recon_mean) / recon_std
    ls = np.abs(lat_norm - lat_mean) / lat_std
    return (1.0 - LATENT_WEIGHT) * rs + LATENT_WEIGHT * ls


def train_stage1_classifier(model, train_loader, val_loader, class_weights, epochs, patience):
    crit = FocalLoss(weight=class_weights, gamma=2.0, label_smoothing=0.05)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)
    best_vl = float("inf")
    best_state = None
    bad = 0
    hist_train, hist_val = [], []
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
        hist_train.append(tr_loss)
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
    return time.time() - t0, hist_train, hist_val


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


# ===============================
# Main pipeline (runs on execute)
# ===============================
if not os.path.isdir(data_path):
    print("ERROR: Dataset not found at:", repr(data_path))
    print("Edit data_path above or place Car-Hacking dataset there.")
else:
    # ===============================
    # Load combined dataframe + 16 features
    # ===============================
    full_df, features = load_full_dataframe(data_path, USE_SUBSET, MAX_NORMAL, MAX_PER_ATTACK_FILE)
    X = full_df[features].values
    y_str = full_df["Label"].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y_str)
    num_classes = len(le.classes_)
    normal_idx = int(le.transform(["Normal"])[0])
    N_FEATURES = len(features)
    print("N_FEATURES:", N_FEATURES, "| Classes:", list(le.classes_))

    # ===============================
    # Preprocessing — stratified splits + MinMax
    # ===============================
    X_trf, X_test, y_trf, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc
    )
    scaler = MinMaxScaler()
    X_trf_s = scaler.fit_transform(X_trf)
    X_test_s = scaler.transform(X_test)

    X_train, X_cal, y_train, y_cal = train_test_split(
        X_trf_s, y_trf, test_size=STAGE1_VAL_FRAC, random_state=RANDOM_STATE, stratify=y_trf
    )

    X_train_t = torch.FloatTensor(X_train).unsqueeze(1)
    y_train_t = torch.LongTensor(y_train)
    X_cal_t = torch.FloatTensor(X_cal).unsqueeze(1)
    y_cal_t = torch.LongTensor(y_cal)
    X_test_t = torch.FloatTensor(X_test_s).unsqueeze(1)
    y_test_t = torch.LongTensor(y_test)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
    )
    cal_loader = DataLoader(
        TensorDataset(X_cal_t, y_cal_t), batch_size=BATCH_SIZE * 2, shuffle=False, pin_memory=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE * 2, shuffle=False, pin_memory=True
    )

    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = torch.FloatTensor(cw).to(device)

    # ===============================
    # Stage 1 — Train classifier
    # ===============================
    stage1_model = CANForgeStage1(N_FEATURES, num_classes).to(device)
    print("\n--- Stage 1: CANForge-style classifier (Focal + label smoothing) ---")
    t_stage1, hist_s1_tr, hist_s1_val = train_stage1_classifier(
        stage1_model, train_loader, cal_loader, class_weights, STAGE1_EPOCHS, STAGE1_PATIENCE
    )
    print("Stage 1 training time (s):", round(t_stage1, 1))

    y_cal_pred = predict_stage1_labels(stage1_model, cal_loader)
    y_test_pred = predict_stage1_labels(stage1_model, test_loader)

    acc1 = accuracy_score(y_test, y_test_pred)
    _, _, f1w, _ = precision_recall_fscore_support(y_test, y_test_pred, average="weighted")
    print("Stage 1 test accuracy:", round(acc1, 4), "| weighted F1:", round(f1w, 4))
    print("\nClassification report (Stage 1):")
    print(classification_report(y_test, y_test_pred, target_names=le.classes_))

    # ===============================
    # Stage 2 — AAE on normal-only train
    # ===============================
    X_normal_train = X_train[y_train == normal_idx]
    Xn_tr, Xn_val = train_test_split(X_normal_train, test_size=0.12, random_state=RANDOM_STATE)
    X_norm_val_t = torch.FloatTensor(Xn_val).to(device)

    enc = Encoder(N_FEATURES, LATENT_DIM).to(device)
    dec = Decoder(LATENT_DIM, N_FEATURES).to(device)
    disc = LatentDiscriminator(LATENT_DIM).to(device)

    ae_curve = []
    print("\n--- Stage 2: Denoising AAE + latent discriminator ---")
    t_stage2, best_val_mse = train_stage2_aae(enc, dec, disc, Xn_tr, X_norm_val_t, ae_curve)
    print("Stage 2 training time (s):", round(t_stage2, 1), "| best val MSE:", round(best_val_mse, 6))

    # ===============================
    # Threshold — normals that pass Stage 1 (calibration split)
    # ===============================
    cal_normal_mask = y_cal == normal_idx
    passed_s1 = y_cal_pred == normal_idx
    pipeline_normal_mask = cal_normal_mask & passed_s1
    if pipeline_normal_mask.sum() < 50:
        print("Warning: few Stage1-pass normals on cal; using all true cal normals.")
        pipeline_normal_mask = cal_normal_mask

    with torch.no_grad():
        xpn = torch.FloatTensor(X_cal[pipeline_normal_mask]).to(device)
        zpn = enc(xpn)
        rpn = dec(zpn)
        val_recon_err = ((xpn - rpn) ** 2).mean(dim=1).cpu().numpy()
        val_lat = torch.norm(zpn, dim=1).cpu().numpy()

    recon_mean = val_recon_err.mean()
    recon_std = val_recon_err.std() + 1e-8
    lat_mean = val_lat.mean()
    lat_std = val_lat.std() + 1e-8

    val_scores = combined_scores(enc, dec, X_cal[pipeline_normal_mask], recon_mean, recon_std, lat_mean, lat_std)
    threshold = float(np.percentile(val_scores, STAGE2_NORMAL_PERCENTILE))
    print(
        "\nStage 2 threshold ({} pct on cal normals | Stage1=Normal):".format(STAGE2_NORMAL_PERCENTILE),
        round(threshold, 6),
    )

    # ===============================
    # Two-stage decisions (test set)
    # ===============================
    test_scores = combined_scores(enc, dec, X_test_s, recon_mean, recon_std, lat_mean, lat_std)
    pred_names = le.inverse_transform(y_test_pred)

    final_label = []
    for i in range(len(y_test)):
        if y_test_pred[i] != normal_idx:
            final_label.append("Known:{}".format(pred_names[i]))
        elif test_scores[i] > threshold:
            final_label.append("ZeroDay")
        else:
            final_label.append("Normal")
    final_label = np.array(final_label)

    true_is_attack = y_test != normal_idx
    hybrid_is_attack = final_label != "Normal"
    h_acc = accuracy_score(true_is_attack, hybrid_is_attack)
    hp, hr, hf1, _ = precision_recall_fscore_support(true_is_attack, hybrid_is_attack, average="binary")

    tn_mask = y_test == normal_idx
    normal_cleared = (final_label[tn_mask] == "Normal").mean() if tn_mask.any() else 0.0
    normal_fpr = (final_label[tn_mask] != "Normal").mean() if tn_mask.any() else 0.0

    print("\n--- Two-stage summary (test) ---")
    print("Hybrid binary Acc:", round(h_acc, 4), "| P:", round(hp, 4), "| R:", round(hr, 4), "| F1:", round(hf1, 4))
    try:
        auc = roc_auc_score(true_is_attack.astype(int), test_scores)
        print("Score AUC (attack vs normal, raw scores):", round(auc, 4))
    except Exception:
        pass
    print("True Normal -> final Normal (both stages):", "{:.2%}".format(normal_cleared))
    print("True Normal -> flagged (Stage 2 FP):", "{:.2%}".format(normal_fpr))

    print("\nDecision counts:")
    for lab in np.unique(final_label):
        print(" ", lab, ":", int((final_label == lab).sum()))

    cm1 = confusion_matrix(y_test, y_test_pred)
    print("\nStage 1 confusion matrix (rows/cols = class order):")
    print(list(le.classes_))
    print(cm1)

    # ===============================
    # Visualizations
    # ===============================
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 10

    # 1) Stage 1 — train vs val focal proxy loss
    fig, ax = plt.subplots(figsize=(8, 4))
    ep1 = range(1, len(hist_s1_tr) + 1)
    ax.plot(ep1, hist_s1_tr, "b-", label="Train loss")
    ax.plot(ep1, hist_s1_val, "b--", label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Focal loss")
    ax.set_title("Stage 1 — Training / Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 2) Stage 1 — confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        cm1,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Stage 1 — Multi-class Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # 3) Stage 2 — mean AE loss per epoch
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(ae_curve) + 1), ae_curve, color="darkgreen", label="Mean AE recon loss / epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Stage 2 — AAE Reconstruction Term (epoch mean)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 4) Combined anomaly score — Normal vs Attack (test)
    y_bin = (y_test != normal_idx).astype(int)
    sc_normal = test_scores[y_test == normal_idx]
    sc_attack = test_scores[y_test != normal_idx]
    fig, ax = plt.subplots(figsize=(8, 4))
    if len(sc_normal):
        ax.hist(
            sc_normal,
            bins=50,
            alpha=0.55,
            label="Normal",
            color="green",
            density=True,
            edgecolor="black",
            linewidth=0.3,
        )
    if len(sc_attack):
        ax.hist(
            sc_attack,
            bins=50,
            alpha=0.55,
            label="Attack",
            color="red",
            density=True,
            edgecolor="black",
            linewidth=0.3,
        )
    ax.axvline(threshold, color="black", linestyle="--", linewidth=2, label="Threshold")
    ax.set_xlabel("Combined anomaly score")
    ax.set_ylabel("Density")
    ax.set_title("Stage 2 — Score: Normal vs Attack (test, all samples)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\nDone. Flow: Stage1 class -> if Normal then Stage2 score vs threshold -> ZeroDay else Normal.")
