import requests
import pyaudio
import wave
import keyboard
import io
from pydub import AudioSegment
from pydub.playback import play
import time
import threading

# --- 설정 ---
SERVER_URL = "http://localhost:8000"
PROCESS_ENDPOINT = f"{SERVER_URL}/process-audio/"
AUDIO_ENDPOINT = f"{SERVER_URL}/audio/"

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

# 녹음 상태를 관리하기 위한 전역 변수 및 이벤트
is_recording = False
stop_recording_event = threading.Event()

def record_audio_non_blocking():
    """
    백그라운드 스레드에서 오디오를 녹음하고, 녹음된 데이터를 BytesIO 객체로 반환합니다.
    `stop_recording_event`가 설정되면 녹음을 중지합니다.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    frames = []
    
    while not stop_recording_event.is_set():
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = io.BytesIO()
    with wave.open(wf, 'wb') as wave_file:
        wave_file.setnchannels(CHANNELS)
        wave_file.setsampwidth(p.get_sample_size(FORMAT))
        wave_file.setframerate(RATE)
        wave_file.writeframes(b''.join(frames))
    
    wf.seek(0)
    return wf

def play_audio_in_thread(audio_data):
    """
    별도의 스레드에서 오디오를 재생하여 메인 스레드를 차단하지 않습니다.
    """
    try:
        audio = AudioSegment.from_file(audio_data, format="mp3")
        play(audio)
    except Exception as e:
        print(f"오디오 재생 중 스레드에서 오류 발생: {e}")

def play_response_audio(audio_path):
    """
    서버로부터 받은 오디오를 비동기적으로 재생합니다.
    """
    try:
        audio_url = f"{AUDIO_ENDPOINT}{audio_path}"
        print(f"AI 응답 오디오 요청: {audio_url}")
        response = requests.get(audio_url, stream=True)
        
        if response.status_code == 200:
            audio_data = io.BytesIO(response.content)
            # 오디오 재생을 위한 새 스레드 시작
            playback_thread = threading.Thread(target=play_audio_in_thread, args=(audio_data,))
            playback_thread.start()
            print("AI 음성 재생 시작...")
        else:
            print(f"오디오 파일을 가져오는 데 실패했습니다. 상태 코드: {response.status_code}")
            print(f"오류 내용: {response.text}")

    except Exception as e:
        print(f"오디오 재생 준비 중 오류 발생: {e}")

def handle_record_key(e):
    """'r' 키 입력을 처리하여 녹음을 시작/중지하는 콜백 함수."""
    global is_recording
    if e.name == 'r':
        if not is_recording:
            is_recording = True
            stop_recording_event.clear()
            print("녹음 시작...")
        else:
            is_recording = False
            stop_recording_event.set()
            print("녹음 중지 중...")

def main():
    """메인 클라이언트 루프"""
    print("--- AI VTuber 마이크 클라이언트 ---")
    print("\n'r' 키를 눌러 녹음을 시작하고, 다시 눌러 종료하세요.")
    print("'q' 키를 누르면 프로그램이 종료됩니다.")

    # 키보드 이벤트 리스너 설정
    keyboard.on_press(handle_record_key)
    
    recording_thread = None
    audio_data_buffer = None

    while True:
        if stop_recording_event.is_set() and recording_thread and not recording_thread.is_alive():
            # 녹음이 완료되었으면 오디오 데이터를 가져오고 서버로 전송
            audio_data = audio_data_buffer.getvalue()
            recording_thread = None
            audio_data_buffer = None
            stop_recording_event.clear() # 다음 녹음을 위해 이벤트 리셋
            
            if len(audio_data) > RATE: # 최소 0.5초 분량의 데이터가 있는지 확인
                print("서버로 오디오를 전송하여 처리 요청 중...")
                try:
                    files = {'audio_file': ('user_audio.wav', audio_data, 'audio/wav')}
                    response = requests.post(PROCESS_ENDPOINT, files=files, timeout=60)

                    if response.status_code == 200:
                        data = response.json()
                        user_text = data.get("user_text")
                        ai_text = data.get("ai_text")
                        audio_path = data.get("audio_path")

                        print("-" * 30)
                        print(f"나: {user_text}")
                        print(f"AI: {ai_text}")
                        print("-" * 30)

                        if audio_path:
                            play_response_audio(audio_path)
                        elif not user_text and not ai_text:
                            print("인식된 내용이 없어 응답이 없습니다.")
                        else:
                            print("응답 오디오가 없습니다.")
                    else:
                        print(f"서버 오류: {response.status_code} - {response.text}")

                except requests.exceptions.RequestException as e:
                    print(f"\n[오류] 서버 요청 중 문제 발생: {e}")
            else:
                print("녹음된 오디오가 너무 짧습니다.")
            
            print("\n'r' 키를 눌러 녹음을 시작하세요.")

        elif is_recording and not recording_thread:
            # 녹음 시작
            audio_data_buffer = io.BytesIO()
            recording_thread = threading.Thread(target=lambda: audio_data_buffer.write(record_audio_non_blocking().getvalue()))
            recording_thread.start()

        if keyboard.is_pressed('q'):
            print("프로그램을 종료합니다.")
            if is_recording:
                stop_recording_event.set()
                if recording_thread:
                    recording_thread.join()
            keyboard.unhook_all()
            break
            
        time.sleep(0.1)

if __name__ == "__main__":
    main()
