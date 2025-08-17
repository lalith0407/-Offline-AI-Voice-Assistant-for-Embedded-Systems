import algo_tflite_wav_module as ds
import time

paths=["C:/Users/lalit_2idtquy/OneDrive/Desktop/Projects/Rasberry Pi/audio_test/on.wav","C:/Users/lalit_2idtquy/OneDrive/Desktop/Projects/Rasberry Pi/audio_test/stop_test.wav"]
prev_time=None
for ele in paths:
    ds.USE_WAV_FILE = True
    ds.WAV_FILE_PATH = ele
    ds.load_wav_audio(ds.WAV_FILE_PATH)
    print("auechuiweahc")
    if not prev_time:
        output = ds.process_audio(mode="wake")
        if output=="on":
            prev_time=time.time()
            print("Listening.......")
    elif prev_time:
        if time.time() - prev_time < 10:
            output = ds.process_audio(mode="command")
            print(output)
        else:
            prev_time=None
        



