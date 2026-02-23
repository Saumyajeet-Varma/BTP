# ===============================
# Stage 2 only: Binary classification (Normal vs Attack)
# Dataset: https://ocslab.hksecurity.net/Datasets/car-hacking-dataset
# ===============================

# ===============================
# Google Drive Mount
# ===============================
from google.colab import drive
drive.mount('/content/drive')

import os
import re
import numpy as np
import pandas as pd

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ===============================
# Dataset Path
# ===============================
data_path = '/content/drive/MyDrive/dataset/9) Car-Hacking Dataset'

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
        return {
            'Timestamp': timestamp,
            'CAN_ID': can_id,
            'DLC': dlc,
            'DATA': data
        }
    return None

file_path = os.path.join(data_path, 'normal_run_data/normal_run_data.txt')
data = []
with open(file_path, 'r') as f:
    for line in f:
        parsed = parse_line(line)
        if parsed:
            data.append(parsed)

df_normal = pd.DataFrame(data)
for i in range(8):
    df_normal[f'DATA{i}'] = df_normal['DATA'].apply(lambda x: x[i] if i < len(x) else 0)
df_normal.drop(columns=['DATA'], inplace=True)
df_normal['Label'] = 'Normal'

# ===============================
# Load Attack CSV Files
# ===============================
column_names = [
    'Timestamp', 'CAN_ID', 'DLC',
    'DATA0', 'DATA1', 'DATA2', 'DATA3', 'DATA4', 'DATA5', 'DATA6', 'DATA7',
    'Flag'
]
df_dos = pd.read_csv(os.path.join(data_path, 'dos_attack.csv'), header=None, names=column_names)
df_dos['Label'] = 'DoS'
df_fuzzy = pd.read_csv(os.path.join(data_path, 'fuzzy_attack.csv'), header=None, names=column_names)
df_fuzzy['Label'] = 'Fuzzy'
df_gear = pd.read_csv(os.path.join(data_path, 'gear_spoofing.csv'), header=None, names=column_names)
df_gear['Label'] = 'Gear'
df_rpm = pd.read_csv(os.path.join(data_path, 'rpm_spoofing.csv'), header=None, names=column_names)
df_rpm['Label'] = 'RPM'

# ===============================
# Combine and Binary Labels (Normal vs Attack)
# ===============================
full_df = pd.concat([df_normal, df_dos, df_fuzzy, df_gear, df_rpm], ignore_index=True)
full_df['BinaryLabel'] = full_df['Label'].apply(lambda x: 'Attack' if x != 'Normal' else 'Normal')

features = ['CAN_ID', 'DLC', 'DATA0', 'DATA1', 'DATA2', 'DATA3', 'DATA4', 'DATA5', 'DATA6', 'DATA7']
N_FEATURES = len(features)
X = full_df[features].values
y_binary = (full_df['BinaryLabel'] == 'Attack').astype(int)  # 0 = Normal, 1 = Attack

# ===============================
# Preprocessing
# ===============================
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=RANDOM_STATE, stratify=y_binary
)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Normal-only subset for Stage 2 (autoencoder training)
X_train_normal = X_train[y_train == 0]
input_dim = X_train_normal.shape[1]

# ===============================
# Stage 2: Autoencoder (Normal vs Attack)
# ===============================
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(RANDOM_STATE)

input_ae = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_ae)
encoded = BatchNormalization()(encoded)
encoded = Dense(32, activation='relu')(encoded)
encoded = Dense(16, activation='relu')(encoded)
decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(input_ae, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

history_autoencoder = autoencoder.fit(
    X_train_normal, X_train_normal,
    epochs=30,
    batch_size=64,
    validation_split=0.1,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)

# ===============================
# Threshold Calculation
# ===============================
reconstructions = autoencoder.predict(X_train_normal, verbose=0)
train_loss = np.mean(np.square(X_train_normal - reconstructions), axis=1)
threshold = np.mean(train_loss) + 3 * np.std(train_loss)
print("Reconstruction threshold (mean + 3×std):", threshold)

# ===============================
# Binary Prediction (Normal vs Attack)
# ===============================
def predict_normal_or_attack(sample, scaled=True):
    """Predict 'Normal' or 'Attack' from reconstruction MSE vs threshold."""
    if not scaled:
        sample = scaler.transform(sample.reshape(1, -1))
    else:
        sample = np.asarray(sample).reshape(1, -1)
    recon = autoencoder.predict(sample, verbose=0)
    loss = np.mean(np.square(sample - recon))
    return "Attack" if loss > threshold else "Normal"

# ===============================
# Evaluation (Binary: Normal vs Attack)
# ===============================
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = np.array([predict_normal_or_attack(X_test[i], scaled=True) for i in range(len(X_test))])
y_true = np.array(['Attack' if y == 1 else 'Normal' for y in y_test])

accuracy = accuracy_score(y_true, y_pred)
print("\n--- Stage 2 (Binary: Normal vs Attack) ---")
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, labels=['Normal', 'Attack']))
print("Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred, labels=['Normal', 'Attack'])
print(cm)

# ===============================
# Visualizations
# ===============================
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# 1. Autoencoder training curve
fig, ax = plt.subplots(figsize=(8, 4))
epochs_ae = range(1, len(history_autoencoder.history['loss']) + 1)
ax.plot(epochs_ae, history_autoencoder.history['loss'], 'b-', label='Train loss')
ax.plot(epochs_ae, history_autoencoder.history['val_loss'], 'b--', label='Val loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('Stage 2 — Autoencoder Reconstruction Loss')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 2. Confusion matrix (Normal vs Attack)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'],
            cbar_kws={'label': 'Count'})
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Stage 2 — Binary Confusion Matrix (Normal vs Attack)')
plt.tight_layout()
plt.show()

# 3. Reconstruction error: Normal vs Attack (test set)
X_test_normal = X_test[y_test == 0]
X_test_attack = X_test[y_test == 1]
recon_normal = autoencoder.predict(X_test_normal, verbose=0)
recon_attack = autoencoder.predict(X_test_attack, verbose=0)
loss_normal = np.mean(np.square(X_test_normal - recon_normal), axis=1)
loss_attack = np.mean(np.square(X_test_attack - recon_attack), axis=1)

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(loss_normal, bins=50, alpha=0.6, label='Normal', color='green', density=True, edgecolor='black', linewidth=0.3)
ax.hist(loss_attack, bins=50, alpha=0.6, label='Attack', color='red', density=True, edgecolor='black', linewidth=0.3)
ax.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
ax.set_xlabel('Reconstruction MSE')
ax.set_ylabel('Density')
ax.set_title('Stage 2 — Reconstruction Error: Normal vs Attack (Test Set)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
