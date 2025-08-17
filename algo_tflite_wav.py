import numpy as np
import tensorflow as tf
import tensorflow.lite as tflite
import scipy.io.wavfile as wav
import sys



MODEL_PATH = "ds_cnn.tflite"
CUSTOM_LABELS = ['silence', 'unknown', 'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
SAMPLE_RATE = 16000
DURATION = 1  
WAKE_WORD = "on"
WAKE_THRESHOLD = 0.97
COMMAND_THRESHOLD = 0.80
ENERGY_THRESHOLD = 0.01
USE_WAV_FILE = False
WAV_FILE_PATH = ""
wav_audio_chunks = []  

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_audio(audio):
    audio = tf.convert_to_tensor(audio, dtype=tf.float32)

    if tf.shape(audio)[0] < SAMPLE_RATE:
        padding = SAMPLE_RATE - tf.shape(audio)[0]
        audio = tf.pad(audio, paddings=[[0, padding]])
    else:
        audio = audio[:SAMPLE_RATE]

    frame_length = 640
    frame_step = 320
    fft_length = 1024

    stft = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
    spectrogram = tf.abs(stft)

    num_mel_bins = 40
    lower_edge_hertz, upper_edge_hertz = 20.0, 4000.0
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, spectrogram.shape[-1], SAMPLE_RATE, lower_edge_hertz, upper_edge_hertz
    )

    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    mfccs = mfccs[..., :10]
    mfccs = tf.expand_dims(mfccs, axis=-1)

    return mfccs


def predict_tflite(mfcc):
    mfcc = tf.expand_dims(mfcc, axis=0)
    interpreter.set_tensor(input_details[0]['index'], mfcc.numpy())
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output


def load_wav_audio(filepath):
    global wav_audio_chunks
    rate, audio = wav.read(filepath)
    if rate != SAMPLE_RATE:
        raise ValueError(f"Expected sample rate {SAMPLE_RATE}, got {rate}")
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max

    chunk_size = SAMPLE_RATE
    wav_audio_chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]

def record_audio(duration=1):
    if USE_WAV_FILE:
        if wav_audio_chunks:
            return wav_audio_chunks.pop(0)
        else:
            exit(0)
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio)


def calculate_energy(audio):
    return np.sqrt(np.mean(np.square(audio)))

def process_audio(mode="wake"):
    audio = record_audio(DURATION)
    energy = calculate_energy(audio)

    if energy < ENERGY_THRESHOLD:
        return None

    mfcc = preprocess_audio(audio)
    pred = predict_tflite(mfcc)[0]
    pred_label_idx = np.argmax(pred)
    pred_label = CUSTOM_LABELS[pred_label_idx]
    pred_confidence = pred[pred_label_idx]

    if mode == "wake":
        if pred_label == WAKE_WORD and pred_confidence > WAKE_THRESHOLD:
            return pred_label
    elif mode == "command":
        if pred_label not in ["on", "silence", "unknown"] and pred_confidence > COMMAND_THRESHOLD:
            return pred_label

    return None


if __name__ == "__main__":
    if len(sys.argv) > 1:
        USE_WAV_FILE = True
        WAV_FILE_PATH = sys.argv[1]
        load_wav_audio(WAV_FILE_PATH)

    if len(sys.argv) > 2:
        mode = sys.argv[2].lower()
    else:
        mode = "wake"

    while True:
        result = process_audio(mode)
        if result:
            print(result)
            
        
