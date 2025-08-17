import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import pathlib

# === SETTINGS ===
DATASET_PATH = r'C:\Users\lalit_2idtquy\Downloads\new\speech_commands_v0.02.tar\speech_commands_v0.02'  # 🔁 Make sure this is where all your class folders are
WANTED_WORDS = ['yes', 'no', 'stop', 'go','silence'] 
BATCH_SIZE = 64

EPOCHS = 20
SAMPLE_RATE = 16000

# === HELPERS ===
def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=1)
    return tf.squeeze(audio, axis=-1)

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]

def get_waveform_and_label(file_path):
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    label = get_label(file_path)
    return waveform, label

def get_spectrogram(waveform):
    zero_padding = tf.zeros([SAMPLE_RATE] - tf.shape(waveform), dtype=tf.float32)
    waveform = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, -1)
    return spectrogram

def preprocess(file_path):
    waveform, label = get_waveform_and_label(file_path)
    spectrogram = get_spectrogram(waveform)
    return spectrogram, label

# === LOAD PATHS ===
data_dir = pathlib.Path(DATASET_PATH)
all_audio_paths = []

for word in WANTED_WORDS:
    word_dir = data_dir / word
    if word_dir.exists():
        word_files = list(word_dir.glob("*.wav"))
        all_audio_paths.extend([str(p) for p in word_files])

# === LABEL ENCODING ===
label_to_index = {label: i for i, label in enumerate(WANTED_WORDS)}

def label_to_number(label):
    return tf.cast(label_to_index[label.numpy().decode()], tf.int64)

def encode_label(spectrogram, label):
    label = tf.py_function(func=label_to_number, inp=[label], Tout=tf.int64)
    label.set_shape([])  # Important: set shape for compatibility
    return spectrogram, label

# === BUILD DATASET ===
files_ds = tf.data.Dataset.from_tensor_slices(tf.constant(all_audio_paths))
spectrogram_ds = files_ds.map(preprocess).map(encode_label).shuffle(1000).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

# === MODEL ===
def create_model(input_shape, num_labels):
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.DepthwiseConv2D((3,3), activation='relu'),
        layers.Conv2D(64, (1,1), activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(num_labels, activation='softmax')
    ])
    return model

# === INPUT SHAPE ===
for spectrogram, _ in spectrogram_ds.take(1):
    input_shape = spectrogram.shape[1:]

model = create_model(input_shape, len(WANTED_WORDS))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# === TRAIN ===
early_stop = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
model.fit(spectrogram_ds, epochs=EPOCHS, callbacks=[early_stop])

# === SAVE ===
model.save("ds_cnn_command_model.h5")
print("✅ Model saved as ds_cnn_command_model.h5")
