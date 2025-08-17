import os
import soundfile as sf
import numpy as np
from scipy.io.wavfile import write

BASE_DIR = r'C:\Users\lalit_2idtquy\Downloads\new\speech_commands_v0.02.tar\speech_commands_v0.02'  # 🔁 Make sure this is where all your class folders are
TARGET_SR = 16000

def convert_to_pcm16(path):
    try:
        data, sr = sf.read(path)
        if data.dtype != np.int16:
            # Convert float32 to int16
            data = np.clip(data, -1.0, 1.0)
            data = (data * 32767).astype(np.int16)
        if sr != TARGET_SR:
            print(f"⚠️ Skipping {path}: wrong sample rate ({sr})")
            return
        write(path, TARGET_SR, data)
        print(f"✅ Converted {path}")
    except Exception as e:
        print(f"❌ Failed to convert {path}: {e}")

for root, _, files in os.walk(BASE_DIR):
    for file in files:
        if file.endswith(".wav"):
            convert_to_pcm16(os.path.join(root, file))
