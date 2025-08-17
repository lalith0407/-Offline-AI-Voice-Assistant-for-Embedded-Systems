import os
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras.models import load_model

# ========== CONFIG ==========
DATASET_PATH = r'C:\Users\lalit_2idtquy\Downloads\new\speech_commands_v0.02.tar\speech_commands_v0.02'
MODEL_PATH = "ds_cnn.h5"
SAMPLE_RATE = 16000
CUSTOM_LABELS = ['yes', 'no', 'stop', 'go', 'marvin', 'unknown', 'silence']
BATCH_SIZE = 32
EPOCHS = 10
AUTOTUNE = tf.data.AUTOTUNE

# ========== AUDIO PREPROCESSING ==========
def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    audio = tf.squeeze(audio, axis=-1)

    # If waveform is too short, pad it
    audio_len = tf.shape(audio)[0]
    needed_len = SAMPLE_RATE

    def pad():
        padding = tf.zeros([needed_len - audio_len], tf.float32)
        return tf.concat([audio, padding], 0)

    def trim():
        return audio[:needed_len]

    audio = tf.cond(audio_len < needed_len, pad, trim)
    return audio


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
    one_hot = label == tf.constant(CUSTOM_LABELS[:-2])  # exclude unknown and silence here
    return mfcc, tf.argmax(one_hot)

def preprocess_unknown(file_path):
    waveform, _ = get_waveform_and_label(file_path)
    mfcc = get_mfcc(waveform)
    label_idx = tf.constant(CUSTOM_LABELS.index('unknown'), dtype=tf.int64)  # <-- Force int64
    return mfcc, label_idx

def preprocess_silence(file_path):
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    waveform = tf.image.random_crop(waveform, size=[SAMPLE_RATE])
    mfcc = get_mfcc(waveform)
    label_idx = tf.constant(CUSTOM_LABELS.index('silence'), dtype=tf.int64)  # <-- Force int64
    return mfcc, label_idx


def is_desired_label(file_path):
    label = tf.strings.split(file_path, os.path.sep)[-2]
    return tf.reduce_any(label == tf.constant(CUSTOM_LABELS[:-2]))  # without unknown and silence

# ========== LOAD DATA ==========
# Main classes
dataset_files = tf.data.Dataset.list_files(str(DATASET_PATH + '/*/*.wav'), shuffle=True)
dataset_files = dataset_files.filter(is_desired_label)
ds_main = dataset_files.map(preprocess, num_parallel_calls=AUTOTUNE)

# Unknown class
all_labels = np.array(tf.io.gfile.listdir(DATASET_PATH))
unknown_labels = [l for l in all_labels if l not in CUSTOM_LABELS and l != '_background_noise_']
unknown_files = []
for label in unknown_labels:
    unknown_files.extend(tf.io.gfile.glob(os.path.join(DATASET_PATH, label, '*.wav')))
unknown_files = tf.data.Dataset.from_tensor_slices(unknown_files)
unknown_ds = unknown_files.map(preprocess_unknown, num_parallel_calls=AUTOTUNE)

# Silence class
background_files = tf.io.gfile.glob(os.path.join(DATASET_PATH, '_background_noise_', '*.wav'))
background_ds = tf.data.Dataset.from_tensor_slices(background_files)
background_ds = background_ds.map(preprocess_silence, num_parallel_calls=AUTOTUNE)

# Merge datasets
final_ds = ds_main.concatenate(unknown_ds).concatenate(background_ds)
final_ds = final_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)

# ========== LOAD & MODIFY MODEL ==========
model = load_model(MODEL_PATH)

# Freeze all layers except last
for layer in model.layers[:-1]:
    layer.trainable = False

# Replace final dense layer if needed
if model.output_shape[-1] != len(CUSTOM_LABELS):
    print(f"Replacing final layer: {model.output_shape[-1]} → {len(CUSTOM_LABELS)}")
    from tensorflow.keras import layers, Model
    base = Model(inputs=model.input, outputs=model.layers[-2].output)
    new_output = layers.Dense(len(CUSTOM_LABELS), activation='softmax')(base.output)
    model = Model(inputs=base.input, outputs=new_output)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ========== TRAIN ==========
model.fit(final_ds, epochs=EPOCHS)

# ========== SAVE ==========
model.save("finetuned_ds_cnn_with_unknown_silence.h5")
print("Model saved as 'finetuned_ds_cnn_with_unknown_silence.h5'")
