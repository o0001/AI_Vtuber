import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os
import time
import httpx # requests 대신 httpx 사용
import torch
import threading
import queue
import asyncio
from utils.logger_config import logger

# 서버 설정
PROCESSING_SERVER_ENDPOINT = config_manager.get('servers', 'processing_server_endpoint')

# 오디오 설정
SAMPLE_RATE = config_manager.getint('audio', 'sample_rate')
CHUNK_SIZE = config_manager.getint('audio', 'chunk_size')
SILENCE_THRESHOLD_SECONDS = config_manager.getfloat('audio', 'silence_threshold_seconds')
MIN_RECORDING_SECONDS = config_manager.getfloat('audio', 'min_recording_seconds')
VAD_THRESHOLD = config_manager.getfloat('audio', 'vad_threshold')


async def send_audio_to_server(filename: str):
    """녹음된 WAV 파일을 처리 서버로 비동기적으로 전송합니다."""
    async with httpx.AsyncClient() as client:
        try:
            with open(filename, 'rb') as f:
                files = {'audio_file': (os.path.basename(filename), f, 'audio/wav')}
                response = await client.post(PROCESSING_SERVER_ENDPOINT, files=files, timeout=10.0)
                response.raise_for_status()
                logger.info(f"오디오 처리 서버 응답: {response.json()['status']}")
        except httpx.ConnectError:
            logger.error(f"오류: 오디오 처리 서버({PROCESSING_SERVER_ENDPOINT})에 연결할 수 없습니다.")
        except httpx.RequestError as e:
            logger.error(f"오디오 전송 중 오류 발생: {e}")
        finally:
            if os.path.exists(filename):
                os.remove(filename)

def save_as_wav(audio_data, filename):
    """녹음된 오디오 데이터를 WAV 파일로 저장합니다."""
    if audio_data is None or len(audio_data) == 0:
        return
    audio_int16 = np.int16(audio_data * 32767)
    write(filename, SAMPLE_RATE, audio_int16)

def sending_thread_func(audio_queue: queue.Queue):
    """오디오 큐에서 데이터를 가져와 파일로 저장하고 서버로 전송하는 스레드"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    while True:
        try:
            recorded_audio = audio_queue.get()
            if recorded_audio is None:
                break

            logger.info("처리 대기열에서 음성 데이터 확인. 전송 준비 중...")
            temp_wav_file = f"temp_capture_{int(time.time())}_{threading.get_ident()}.wav"
            save_as_wav(recorded_audio, temp_wav_file)
            
            loop.run_until_complete(send_audio_to_server(temp_wav_file))
            
            audio_queue.task_done()
        except Exception as e:
            logger.error(f"전송 스레드 오류: {e}")

def recording_thread_func(audio_queue: queue.Queue, model_vad, utils):
    """마이크 입력을 감지하고 음성 구간을 큐에 넣는 스레드"""
    (get_speech_timestamps, _, _, VADIterator, _) = utils
    vad_iterator = VADIterator(model_vad, threshold=VAD_THRESHOLD)
    
    current_speech_segment_buffer = []
    is_speech_active = False
    last_speech_activity_time = time.time()

    logger.info("\n음성 활동을 감지 중입니다. 말씀해주세요...")

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=CHUNK_SIZE) as stream:
            while True:
                audio_chunk, overflowed = stream.read(CHUNK_SIZE)
                if overflowed:
                    logger.warning("경고: 오디오 버퍼 오버플로우 발생!")

                audio_chunk_np = audio_chunk.flatten()
                vad_result = vad_iterator(audio_chunk_np, return_seconds=True)

                def process_and_queue_audio():
                    nonlocal current_speech_segment_buffer
                    if not current_speech_segment_buffer:
                        return
                    
                    recorded_audio = np.concatenate(current_speech_segment_buffer, axis=0)
                    duration = len(recorded_audio) / SAMPLE_RATE

                    if duration < MIN_RECORDING_SECONDS:
                        logger.info(f"녹음된 음성이 너무 짧습니다({duration:.2f}초). 무시합니다.")
                    else:
                        logger.info(f"음성 구간 종료 감지({duration:.2f}초). 처리 대기열에 추가...")
                        audio_queue.put(recorded_audio)
                    
                    current_speech_segment_buffer = []

                if vad_result is not None:
                    if 'start' in vad_result:
                        if not is_speech_active:
                            logger.info("음성 구간 시작 감지. 녹음 시작...")
                            is_speech_active = True
                            current_speech_segment_buffer = []
                        current_speech_segment_buffer.append(audio_chunk)
                        last_speech_activity_time = time.time()
                    elif 'end' in vad_result:
                        if is_speech_active:
                            is_speech_active = False
                            current_speech_segment_buffer.append(audio_chunk)
                            process_and_queue_audio()
                        last_speech_activity_time = time.time()
                
                elif is_speech_active:
                    current_speech_segment_buffer.append(audio_chunk)
                    if time.time() - last_speech_activity_time > SILENCE_THRESHOLD_SECONDS:
                        logger.info("장시간 침묵 감지. 음성 구간 강제 종료...")
                        is_speech_active = False
                        process_and_queue_audio()
                        last_speech_activity_time = time.time()

    except Exception as e:
        logger.error(f"녹음 스레드 오류: {e}")

def main():
    """메인 실행 함수"""
    audio_queue = queue.Queue()

    logger.info("Silero VAD 모델을 로드합니다...")
    try:
        model_vad, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                          model='silero_vad',
                                          force_reload=False)
        logger.info("Silero VAD 모델 로드 완료.")
    except Exception as e:
        logger.error(f"Silero VAD 모델 로드 중 오류 발생: {e}")
        return

    sender_thread = threading.Thread(target=sending_thread_func, args=(audio_queue,), daemon=True)
    sender_thread.start()

    try:
        recording_thread_func(audio_queue, model_vad, utils)
    except KeyboardInterrupt:
        logger.info("\n프로그램을 종료합니다.")
        audio_queue.put(None)
        sender_thread.join()

if __name__ == "__main__":
    main()


