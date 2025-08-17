import os
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras.models import load_model

# ========== CONFIG ==========
DATASET_PATH = r'C:\Users\lalit_2idtquy\Downloads\new\speech_commands_v0.02.tar\speech_commands_v0.02'  # 🔁 Make sure this is where all your class folders are
MODEL_PATH = "ds_cnn.h5"
SAMPLE_RATE = 16000
CUSTOM_LABELS = ['yes', 'no', 'stop','go']  # <-- Your desired classes
BATCH_SIZE = 32
EPOCHS = 10
AUTOTUNE = tf.data.AUTOTUNE

# ========== AUDIO PREPROCESSING ==========
def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    audio = tf.squeeze(audio, axis=-1)
    padding = tf.zeros([SAMPLE_RATE] - tf.shape(audio), tf.float32)
    audio = tf.concat([audio, padding], 0)
    return audio[:SAMPLE_RATE]

def get_label(file_path):
    return tf.strings.split(file_path, os.path.sep)[-2]

def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label

def get_mfcc(waveform):
    stft = tf.signal.stft(waveform, frame_length=640, frame_step=320)
    spectrogram = tf.abs(stft)
    mel_spectrogram = tf.tensordot(spectrogram, tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=40,
        num_spectrogram_bins=spectrogram.shape[-1],
        sample_rate=SAMPLE_RATE
    ), 1)
    log_mel = tf.math.log(mel_spectrogram + 1e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)[..., :10]
    return mfccs

def preprocess(file_path):
    waveform, label = get_waveform_and_label(file_path)
    mfcc = get_mfcc(waveform)
    one_hot = label == tf.constant(CUSTOM_LABELS)
    return mfcc, tf.argmax(one_hot)

def is_desired_label(file_path):
    label = tf.strings.split(file_path, os.path.sep)[-2]
    return tf.reduce_any(label == tf.constant(CUSTOM_LABELS))

# ========== LOAD DATA ==========
files = tf.data.Dataset.list_files(str(DATASET_PATH + '/*/*.wav'), shuffle=True)
files = files.filter(is_desired_label)
ds = files.map(preprocess, num_parallel_calls=AUTOTUNE)
ds = ds.shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)

# ========== LOAD & MODIFY MODEL ==========
model = load_model(MODEL_PATH)

# Optionally freeze all layers except the last dense layer
for layer in model.layers[:-1]:
    layer.trainable = False

# Replace final dense layer if label count doesn't match
if model.output_shape[-1] != len(CUSTOM_LABELS):
    print(f"Replacing final layer: {model.output_shape[-1]} → {len(CUSTOM_LABELS)}")
    from tensorflow.keras import layers, Model

    base = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    new_output = layers.Dense(len(CUSTOM_LABELS), activation='softmax')(base.output)
    model = Model(inputs=base.input, outputs=new_output)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ========== TRAIN ==========
model.fit(ds, epochs=EPOCHS)

# ========== SAVE ==========
model.save("finetuned_ds_cnn_subset_with_background_1.h5")
print("Model saved as 'finetuned_ds_cnn_subset_with_background.h5'")
