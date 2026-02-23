# ===============================
# Dataset Link: https://ocslab.hksecurity.net/Datasets/car-hacking-dataset
# ===============================
# REVAMP PIPELINE: Normal-first detection, then attack-type classification.
# Stage 1: Is the message Normal or Not? (Autoencoder on normal only.)
# Stage 2: If Not Normal → classify into DoS, Fuzzy, Gear, RPM.
# ===============================

# ===============================
# Google Drive Mount
# ===============================
from google.colab import drive
drive.mount('/content/drive')

import os
import pandas as pd
import numpy as np
import re

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
    'DATA0','DATA1','DATA2','DATA3','DATA4','DATA5','DATA6','DATA7',
    'Flag'
]

def convert_numeric_columns(df, columns_to_convert):
    for col in columns_to_convert:
        if col == 'CAN_ID' or col.startswith('DATA'):
            # Convert from hex string to integer, handling potential non-string values
            # Use regex to check if string looks like a hex number before conversion
            df[col] = df[col].astype(str).apply(lambda x: int(x, 16) if re.match(r'^[0-9a-fA-F]+$', x.strip()) else np.nan)
        else:
            # For other numeric columns like DLC, convert directly to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # Fill any NaN values with 0 and convert to integer type
        df[col] = df[col].fillna(0).astype(int)
    return df

cols_to_process = ['CAN_ID', 'DLC'] + [f'DATA{i}' for i in range(8)]

# Flag in attack CSVs: 'T' = injected (attacked), 'R' = normal message
def label_from_flag(flag_series, attack_name):
    """Label each row: Normal if Flag=='R', else attack type if Flag=='T'."""
    labels = []
    for v in flag_series.astype(str).str.strip().str.upper():
        if v == 'R':
            labels.append('Normal')
        elif v == 'T':
            labels.append(attack_name)
        else:
            # Fallback: treat unknown as Normal to be conservative
            labels.append('Normal')
    return labels

df_dos = pd.read_csv(os.path.join(data_path,'DoS_dataset.csv'),
                     header=None, names=column_names)
df_dos = convert_numeric_columns(df_dos, cols_to_process)
df_dos['Label'] = label_from_flag(df_dos['Flag'], 'DoS')
print("DoS df done")

df_fuzzy = pd.read_csv(os.path.join(data_path,'Fuzzy_dataset.csv'),
                       header=None, names=column_names)
df_fuzzy = convert_numeric_columns(df_fuzzy, cols_to_process)
df_fuzzy['Label'] = label_from_flag(df_fuzzy['Flag'], 'Fuzzy')
print("Fuzzy df done")

df_gear = pd.read_csv(os.path.join(data_path,'gear_dataset.csv'),
                      header=None, names=column_names)
df_gear = convert_numeric_columns(df_gear, cols_to_process)
df_gear['Label'] = label_from_flag(df_gear['Flag'], 'Gear')
print("gear df done")

df_rpm = pd.read_csv(os.path.join(data_path,'RPM_dataset.csv'),
                     header=None, names=column_names)
df_rpm = convert_numeric_columns(df_rpm, cols_to_process)
df_rpm['Label'] = label_from_flag(df_rpm['Flag'], 'RPM')
print("RPM df done")

# ===============================
# Combine All Data
# ===============================
full_df = pd.concat([df_normal, df_dos, df_fuzzy, df_gear, df_rpm], ignore_index=True)

features = ['CAN_ID', 'DLC', 'DATA0', 'DATA1', 'DATA2', 'DATA3', 'DATA4', 'DATA5', 'DATA6', 'DATA7']

# ===============================
# Exploratory Data Analysis (EDA) — same as cursor
# ===============================
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

fig, ax = plt.subplots(figsize=(8, 4))
counts = full_df['Label'].value_counts()
colors = sns.color_palette("Set2", n_colors=len(counts))
bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Class')
ax.set_ylabel('Number of messages')
ax.set_title('EDA — Class distribution (Revamp pipeline)')
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts.values)*0.008,
            f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=9)
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()

N_FEATURES = len(features)
X = full_df[features].values
y = full_df['Label'].values

# ===============================
# Preprocessing
# ===============================
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Full label encoder (all 5 classes) for final evaluation
le_full = LabelEncoder()
y_encoded_full = le_full.fit_transform(y)
y_cat_full = to_categorical(y_encoded_full)

# Split: stratify by full label
X_train, X_test, y_train_full, y_test_full = train_test_split(
    X, y_cat_full, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded_full
)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_lstm = X_train.reshape((X_train.shape[0], 1, N_FEATURES))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, N_FEATURES))

# ===============================
# Stage 1: Autoencoder — Normal vs Not Normal
# ===============================
# Train only on normal samples. High reconstruction error → Not Normal.
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(RANDOM_STATE)

normal_label_full = le_full.transform(['Normal'])[0]
X_train_normal = X_train[np.argmax(y_train_full, axis=1) == normal_label_full]

input_dim = X_train_normal.shape[1]

input_ae = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_ae)
encoded = BatchNormalization()(encoded)
encoded = Dense(32, activation='relu')(encoded)
encoded = Dense(16, activation='relu')(encoded)
decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(input_ae, decoded)
autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])

history_autoencoder = autoencoder.fit(
    X_train_normal, X_train_normal,
    epochs=30,
    batch_size=64,
    validation_split=0.1,
    verbose=2,  # one line per epoch: loss, mae, val_loss, val_mae
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)

# ===============================
# Stage 1: Threshold (Normal vs Not Normal)
# ===============================
reconstructions = autoencoder.predict(X_train_normal, verbose=0)
train_loss = np.mean(np.square(X_train_normal - reconstructions), axis=1)
threshold = np.mean(train_loss) + 3 * np.std(train_loss)
print("Stage 1 — Reconstruction threshold (Normal vs Not Normal):", threshold)

# ===============================
# Stage 2: Attack-type classifier (DoS, Fuzzy, Gear, RPM only)
# ===============================
# Train LSTM only on attack samples. Used only when Stage 1 says "Not Normal".
attack_classes = ['DoS', 'Fuzzy', 'Gear', 'RPM']
le_attack = LabelEncoder()
le_attack.fit(attack_classes)

# Mask: train samples that are attacks (no Normal)
train_attack_mask = np.argmax(y_train_full, axis=1) != normal_label_full
X_train_attack = X_train[train_attack_mask]
y_train_attack_labels = np.argmax(y_train_full, axis=1)[train_attack_mask]
# Map full class indices to attack-only indices (0=DoS, 1=Fuzzy, 2=Gear, 3=RPM)
y_train_attack = le_attack.transform(le_full.inverse_transform(y_train_attack_labels))
y_train_attack_cat = to_categorical(y_train_attack, num_classes=len(attack_classes))

X_train_attack_lstm = X_train_attack.reshape((X_train_attack.shape[0], 1, N_FEATURES))

from tensorflow.keras.layers import LSTM

inputs = Input(shape=(1, N_FEATURES))
x = LSTM(128, return_sequences=False)(inputs)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(32, activation='relu')(x)
outputs = Dense(len(attack_classes), activation='softmax')(x)

attack_classifier = Model(inputs, outputs)
attack_classifier.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

from sklearn.utils.class_weight import compute_class_weight
classes_attack = np.unique(y_train_attack)
class_weights_attack = compute_class_weight(
    'balanced', classes=classes_attack, y=y_train_attack
)
class_weights_attack_dict = dict(zip(classes_attack, class_weights_attack))

history_attack = attack_classifier.fit(
    X_train_attack_lstm, y_train_attack_cat,
    epochs=30,
    batch_size=64,
    validation_split=0.1,
    verbose=2,  # one line per epoch: loss, accuracy, val_loss, val_accuracy
    class_weight=class_weights_attack_dict,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)

# ===============================
# Revamp prediction: Normal first, then attack type
# ===============================
def revamp_predict(sample, scaled=True):
    """
    Stage 1: Normal vs Not Normal (autoencoder).
    Stage 2: If Not Normal → classify into DoS, Fuzzy, Gear, RPM.
    """
    if not scaled:
        sample = scaler.transform(sample.reshape(1, -1))
    else:
        sample = np.asarray(sample).reshape(1, -1)

    # Stage 1: Reconstruction error
    recon = autoencoder.predict(sample, verbose=0)
    loss = np.mean(np.square(sample - recon))

    if loss <= threshold:
        return "Normal"

    # Stage 2: Attack-type classification
    sample_lstm = sample.reshape((1, 1, N_FEATURES))
    pred = attack_classifier.predict(sample_lstm, verbose=0)
    class_idx = np.argmax(pred)
    return attack_classes[class_idx]

# ===============================
# Evaluation
# ===============================
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

revamp_results = [revamp_predict(X_test[i], scaled=True) for i in range(len(X_test))]
true_labels = le_full.inverse_transform(np.argmax(y_test_full, axis=1))
revamp_accuracy = accuracy_score(true_labels, revamp_results)
print("Revamp pipeline accuracy:", revamp_accuracy)

# ===============================
# Visualizations
# ===============================
all_labels_revamp = list(le_full.classes_)
cm = confusion_matrix(true_labels, revamp_results, labels=all_labels_revamp)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=all_labels_revamp, yticklabels=all_labels_revamp,
            cbar_kws={'label': 'Count'})
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Revamp pipeline — Confusion Matrix (Normal first, then attack type)')
plt.xticks(rotation=25)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Reconstruction error: Normal vs Attack (test)
test_normal_mask = np.argmax(y_test_full, axis=1) == normal_label_full
X_test_normal = X_test[test_normal_mask]
X_test_attack = X_test[~test_normal_mask]
recon_test_normal = autoencoder.predict(X_test_normal, verbose=0)
recon_test_attack = autoencoder.predict(X_test_attack, verbose=0)
loss_normal = np.mean(np.square(X_test_normal - recon_test_normal), axis=1)
loss_attack = np.mean(np.square(X_test_attack - recon_test_attack), axis=1)
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(loss_normal, bins=50, alpha=0.6, label='Normal', color='green', density=True, edgecolor='black', linewidth=0.3)
ax.hist(loss_attack, bins=50, alpha=0.6, label='Attack', color='red', density=True, edgecolor='black', linewidth=0.3)
ax.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
ax.set_xlabel('Reconstruction MSE')
ax.set_ylabel('Density')
ax.set_title('Stage 1 — Reconstruction Error: Normal vs Attack (Test Set)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n--- Revamp pipeline — Classification Report ---")
print(classification_report(true_labels, revamp_results))
