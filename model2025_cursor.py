# ===============================
# Dataset Link: https://ocslab.hksecurity.net/Datasets/car-hacking-dataset
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
        # Pad DATA to 8 bytes (dataset: DLC 0-8, DATA[0-7])
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

df_dos = pd.read_csv(os.path.join(data_path,'dos_attack.csv'),
                     header=None, names=column_names)
df_dos['Label'] = 'DoS'

df_fuzzy = pd.read_csv(os.path.join(data_path,'fuzzy_attack.csv'),
                       header=None, names=column_names)
df_fuzzy['Label'] = 'Fuzzy'

df_gear = pd.read_csv(os.path.join(data_path,'gear_spoofing.csv'),
                      header=None, names=column_names)
df_gear['Label'] = 'Gear'

df_rpm = pd.read_csv(os.path.join(data_path,'rpm_spoofing.csv'),
                     header=None, names=column_names)
df_rpm['Label'] = 'RPM'

# ===============================
# Combine All Data
# ===============================
full_df = pd.concat([df_normal, df_dos, df_fuzzy, df_gear, df_rpm], ignore_index=True)

# Feature columns used for modeling (defined early for EDA)
features = ['CAN_ID', 'DLC', 'DATA0', 'DATA1', 'DATA2', 'DATA3', 'DATA4', 'DATA5', 'DATA6', 'DATA7']

# ===============================
# Exploratory Data Analysis (EDA)
# ===============================
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# ----- EDA 1: Class distribution -----
fig, ax = plt.subplots(figsize=(8, 4))
counts = full_df['Label'].value_counts()
colors = sns.color_palette("Set2", n_colors=len(counts))
bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Class')
ax.set_ylabel('Number of messages')
ax.set_title('EDA — Class distribution')
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts.values)*0.008,
            f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=9)
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()

# ----- EDA 2: Top CAN IDs (overall) -----
fig, ax = plt.subplots(figsize=(10, 4))
top_can = full_df['CAN_ID'].value_counts().head(20)
ax.barh(range(len(top_can)), top_can.values, color=sns.color_palette("viridis", len(top_can)))
ax.set_yticks(range(len(top_can)))
ax.set_yticklabels([f'0x{x:04X}' for x in top_can.index], fontsize=9)
ax.set_xlabel('Message count')
ax.set_ylabel('CAN ID')
ax.set_title('EDA — Top 20 CAN IDs by frequency')
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# ----- EDA 3: DLC distribution -----
fig, ax = plt.subplots(figsize=(7, 4))
dlc_counts = full_df['DLC'].value_counts().sort_index()
ax.bar(dlc_counts.index.astype(str), dlc_counts.values, color='steelblue', edgecolor='black', linewidth=0.5)
ax.set_xlabel('DLC (Data Length Code)')
ax.set_ylabel('Count')
ax.set_title('EDA — DLC distribution (0–8 bytes)')
plt.tight_layout()
plt.show()

# ----- EDA 4: Feature correlation heatmap -----
fig, ax = plt.subplots(figsize=(9, 7))
corr = full_df[features].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax, square=True,
            linewidths=0.5, cbar_kws={'label': 'Correlation'})
ax.set_title('EDA — Correlation between CAN_ID, DLC, DATA0–7')
plt.tight_layout()
plt.show()

# ----- EDA 5: DATA byte distributions by class (box plots, one byte per subplot) -----
data_cols = [f'DATA{i}' for i in range(8)]
fig, axes = plt.subplots(2, 4, figsize=(14, 8))
axes = axes.flatten()
for i, col in enumerate(data_cols):
    sns.boxplot(data=full_df, x='Label', y=col, ax=axes[i], palette='Set3')
    axes[i].set_title(col)
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', rotation=25)
fig.suptitle('EDA — DATA byte distributions by class', fontsize=12, y=1.02)
plt.tight_layout()
plt.show()

# ----- EDA 6: CAN_ID distribution by class (stacked/top IDs per class) -----
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for idx, label in enumerate(full_df['Label'].unique()):
    sub = full_df[full_df['Label'] == label]
    top = sub['CAN_ID'].value_counts().head(10)
    ax = axes[idx]
    ax.barh(range(len(top)), top.values, color=sns.color_palette("rocket", len(top)))
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f'0x{x:04X}' for x in top.index], fontsize=8)
    ax.set_xlabel('Count')
    ax.set_title(label)
    ax.invert_yaxis()
# Hide unused subplot if 5 classes
if len(full_df['Label'].unique()) < 6:
    axes[5].axis('off')
plt.suptitle('EDA — Top 10 CAN IDs per class', fontsize=12, y=1.02)
plt.tight_layout()
plt.show()

# ----- EDA 7: Sample sizes and basic stats table -----
print("\n--- EDA — Dataset summary ---")
print(f"Total messages: {len(full_df):,}")
print(f"Classes: {list(full_df['Label'].unique())}")
print("\nSample count per class:")
print(full_df['Label'].value_counts().to_string())
print("\nBasic stats (features):")
print(full_df[features].describe().round(2).to_string())

N_FEATURES = len(features)
X = full_df[features].values
y = full_df['Label'].values

# ===============================
# Preprocessing
# ===============================
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

# Split first so we fit scaler only on train (avoid leakage; stratify needs 1D labels)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_lstm = X_train.reshape((X_train.shape[0], 1, N_FEATURES))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, N_FEATURES))

# ===============================
# Stage 1: LSTM Classifier
# ===============================
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(RANDOM_STATE)

# Optional class weights for imbalanced CAN attack classes
from sklearn.utils.class_weight import compute_class_weight
train_labels_encoded = np.argmax(y_train, axis=1)
classes = np.unique(train_labels_encoded)
class_weights_arr = compute_class_weight(
    'balanced', classes=classes, y=train_labels_encoded
)
class_weights = dict(zip(classes, class_weights_arr))

inputs = Input(shape=(1, N_FEATURES))
x = LSTM(128, return_sequences=False)(inputs)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(32, activation='relu')(x)
outputs = Dense(len(le.classes_), activation='softmax')(x)

classifier = Model(inputs, outputs)
classifier.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_classifier = classifier.fit(
    X_train_lstm, y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.1,
    class_weight=class_weights,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)

# ===============================
# Stage 2: Autoencoder
# ===============================
normal_label = le.transform(['Normal'])[0]
X_train_normal = X_train[np.argmax(y_train, axis=1) == normal_label]

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
reconstructions = autoencoder.predict(X_train_normal)
train_loss = np.mean(np.square(X_train_normal - reconstructions), axis=1)
threshold = np.mean(train_loss) + 3*np.std(train_loss)

print("Reconstruction Threshold:", threshold)

# ===============================
# Hybrid Prediction Function
# ===============================
def hybrid_predict(sample, scaled=True):
    """
    sample: one row of features (length N_FEATURES).
    scaled: if True, sample is already MinMax-scaled (e.g. from train/test set);
            if False, sample is raw and will be scaled using the fitted scaler.
    """
    if not scaled:
        sample = scaler.transform(sample.reshape(1, -1))
    else:
        sample = np.asarray(sample).reshape(1, -1)
    sample_lstm = sample.reshape((1, 1, N_FEATURES))

    pred = classifier.predict(sample_lstm, verbose=0)
    class_idx = np.argmax(pred)
    class_name = le.inverse_transform([class_idx])[0]

    if class_name != "Normal":
        return class_name

    recon = autoencoder.predict(sample, verbose=0)
    loss = np.mean(np.square(sample - recon))

    if loss > threshold:
        return "Unknown Attack"
    else:
        return "Normal"

# ===============================
# Hybrid Evaluation
# ===============================
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

hybrid_results = []
for i in range(len(X_test)):
    hybrid_results.append(hybrid_predict(X_test[i], scaled=True))

true_labels = le.inverse_transform(np.argmax(y_test, axis=1))
hybrid_accuracy = accuracy_score(true_labels, hybrid_results)
print("Hybrid Accuracy:", hybrid_accuracy)

# ===============================
# Visualizations for Evaluation
# ===============================
import matplotlib.pyplot as plt
import seaborn as sns

# Use a style that works well in notebooks
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# ----- 1. Class distribution in dataset -----
fig, ax = plt.subplots(figsize=(8, 4))
label_counts = full_df['Label'].value_counts()
colors = sns.color_palette("Set2", n_colors=len(label_counts))
bars = ax.bar(label_counts.index, label_counts.values, color=colors, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Class')
ax.set_ylabel('Count')
ax.set_title('Class Distribution in Dataset')
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(label_counts.values)*0.01,
            str(int(bar.get_height())), ha='center', va='bottom', fontsize=9)
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()

# ----- 2. LSTM classifier training curves -----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
epochs_c = range(1, len(history_classifier.history['loss']) + 1)
ax1.plot(epochs_c, history_classifier.history['loss'], 'b-', label='Train loss')
ax1.plot(epochs_c, history_classifier.history['val_loss'], 'b--', label='Val loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('LSTM Classifier — Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax2.plot(epochs_c, history_classifier.history['accuracy'], 'g-', label='Train accuracy')
ax2.plot(epochs_c, history_classifier.history['val_accuracy'], 'g--', label='Val accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('LSTM Classifier — Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ----- 3. Autoencoder training curve -----
fig, ax = plt.subplots(figsize=(8, 4))
epochs_ae = range(1, len(history_autoencoder.history['loss']) + 1)
ax.plot(epochs_ae, history_autoencoder.history['loss'], 'b-', label='Train loss')
ax.plot(epochs_ae, history_autoencoder.history['val_loss'], 'b--', label='Val loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('Autoencoder — Reconstruction Loss')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ----- 4. Confusion matrix (hybrid predictions) -----
# Include all classes (encoder order); "Unknown Attack" may appear only in predictions
all_labels = list(le.classes_) + ["Unknown Attack"]
all_labels = sorted(set(all_labels))
cm = confusion_matrix(true_labels, hybrid_results, labels=all_labels)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=all_labels, yticklabels=all_labels,
            cbar_kws={'label': 'Count'})
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Hybrid Model — Confusion Matrix')
plt.xticks(rotation=25)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# ----- 5. Reconstruction error: Normal vs Attack (test set) -----
normal_label_idx = le.transform(['Normal'])[0]
test_normal_mask = np.argmax(y_test, axis=1) == normal_label_idx
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
ax.set_title('Autoencoder Reconstruction Error — Normal vs Attack (Test Set)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ----- 6. Classification report (text summary) -----
print("\n--- Hybrid Model — Classification Report ---")
print(classification_report(true_labels, hybrid_results))