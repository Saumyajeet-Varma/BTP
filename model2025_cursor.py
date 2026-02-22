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

# CAN_ID, DLC, DATA[0-7] per dataset attributes
features = ['CAN_ID', 'DLC', 'DATA0', 'DATA1', 'DATA2', 'DATA3', 'DATA4', 'DATA5', 'DATA6', 'DATA7']
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

classifier.fit(
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

autoencoder.fit(
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
from sklearn.metrics import accuracy_score

hybrid_results = []
for i in range(len(X_test)):
    hybrid_results.append(hybrid_predict(X_test[i], scaled=True))

true_labels = le.inverse_transform(np.argmax(y_test, axis=1))

print("Hybrid Accuracy:", accuracy_score(true_labels, hybrid_results))