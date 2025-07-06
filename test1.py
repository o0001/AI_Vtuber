import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os
import time
import requests
import torch
import soundfile as sf
import threading
import queue

SAMPLE_RATE = 16000
PROCESSING_SERVER_ENDPOINT = "http://localhost:8001/process_audio/"
SILENCE_THRESHOLD_SECONDS = 1.5
CHUNK_SIZE = 512

def send_audio_to_server(filename: str):
    try:
        with open(filename, 'rb') as f:
            files = {'audio_file': (os.path.basename(filename), f, 'audio/wav')}
            response = requests.post(PROCESSING_SERVER_ENDPOINT, files=files)
            response.raise_for_status()
            print(f"res: {response.json()['status']}")
    except requests.exceptions.ConnectionError:
        print(f"cant({PROCESSING_SERVER_ENDPOINT})conect.")
    except requests.exceptions.RequestException as e:
        print(f"error: {e}")
    finally:
        if os.path.exists(filename):
            os.remove(filename)

def save_as_wav(audio_data, filename):
    if audio_data is None or len(audio_data) == 0:
        return
    audio_int16 = np.int16(audio_data * 32767)
    write(filename, SAMPLE_RATE, audio_int16)

def sending_thread_func(audio_queue: queue.Queue):
    while True:
        try:
            recorded_audio = audio_queue.get()
            if recorded_audio is None:
                break

            print("prepair sending audio to server...")
            temp_wav_file = f"temp_capture_{int(time.time())}_{threading.get_ident()}.wav"
            save_as_wav(recorded_audio, temp_wav_file)
            send_audio_to_server(temp_wav_file)
            
            audio_queue.task_done()
        except Exception as e:
            print(f"error: {e}")

def recording_thread_func(audio_queue: queue.Queue, model_vad, utils):
    (get_speech_timestamps, _, _, VADIterator, _) = utils
    vad_iterator = VADIterator(model_vad, threshold=0.4)
    
    current_speech_segment_buffer = []
    is_speech_active = False
    last_speech_activity_time = time.time()

    print("\nlsn ing")
# 아 시발 뭔가 맘에 안드는데
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=CHUNK_SIZE) as stream:
            while True:
                audio_chunk, overflowed = stream.read(CHUNK_SIZE)
                if overflowed:
                    print("worning: audio buffer overflow detected. Skipping chunk.")

                audio_chunk_np = audio_chunk.flatten()
                vad_result = vad_iterator(audio_chunk_np, return_seconds=True)

                if vad_result is not None:
                    if 'start' in vad_result:
                        if not is_speech_active:
                            print("recording")
                            is_speech_active = True
                            current_speech_segment_buffer = []
                        current_speech_segment_buffer.append(audio_chunk)
                        last_speech_activity_time = time.time()
                    elif 'end' in vad_result:
                        if is_speech_active:
                            print("ending recording")
                            is_speech_active = False
                            current_speech_segment_buffer.append(audio_chunk)
                            
                            if current_speech_segment_buffer:
                                recorded_audio = np.concatenate(current_speech_segment_buffer, axis=0)
                                audio_queue.put(recorded_audio)
                                current_speech_segment_buffer = []
                        last_speech_activity_time = time.time()
                
                elif is_speech_active:
                    current_speech_segment_buffer.append(audio_chunk)
                    if time.time() - last_speech_activity_time > SILENCE_THRESHOLD_SECONDS:
                        print("pffffff")
                        is_speech_active = False
                        
                        if current_speech_segment_buffer:
                            recorded_audio = np.concatenate(current_speech_segment_buffer, axis=0)
                            audio_queue.put(recorded_audio)
                            current_speech_segment_buffer = []
                        last_speech_activity_time = time.time()

    except Exception as e:
        print(f"error: {e}")

def main():
    audio_queue = queue.Queue()

    print("rod vad")
    try:
        model_vad, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                          model='silero_vad',
                                          force_reload=False)
        print("vad se.")
    except Exception as e:
        print(f"error: {e}")
        return

    sender_thread = threading.Thread(target=sending_thread_func, args=(audio_queue,), daemon=True)
    sender_thread.start()

    try:
        recording_thread_func(audio_queue, model_vad, utils)
    except KeyboardInterrupt:
        print("\nout.")
        audio_queue.put(None)
        sender_thread.join()

if __name__ == "__main__":
    main()
