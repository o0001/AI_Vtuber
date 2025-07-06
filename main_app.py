import os
import queue
import threading
import time
import uuid
import numpy as np
import torch
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import whisper

from utils.config_manager import config_manager
from utils.logger_config import logger
from llm_service import generate_response_text, load_llm_model
from audio_service import text_to_speech

# --- 설정 로드 ---
# 오디오 설정
SAMPLE_RATE = config_manager.getint('audio', 'sample_rate')
CHUNK_SIZE = config_manager.getint('audio', 'chunk_size')
SILENCE_THRESHOLD_SECONDS = config_manager.getfloat('audio', 'silence_threshold_seconds')
MIN_RECORDING_SECONDS = config_manager.getfloat('audio', 'min_recording_seconds')
VAD_THRESHOLD = config_manager.getfloat('audio', 'vad_threshold')

# STT 모델
STT_MODEL_TYPE = config_manager.get('models', 'stt_model_type')
STT_LANGUAGE = config_manager.get('stt', 'language')

# 감정 분석(SER) 모델
SER_MODEL_FAST = config_manager.get('models', 'ser_model_fast')
SER_MODEL_PRECISE = config_manager.get('models', 'ser_model_precise')
CONFIDENCE_THRESHOLD = config_manager.getfloat('analysis', 'confidence_threshold')

# 대화 기록
MEMORY_DEPTH = config_manager.getint('llm_params', 'memory_depth', fallback=5)

# --- 전역 변수 및 큐 ---
audio_queue = queue.Queue() # 녹음된 오디오 청크가 들어가는 큐
processing_queue = queue.Queue() # (오디오 경로, 오디오 데이터)가 들어가는 큐
chat_history = []
is_processing = threading.Event() # 현재 AI가 응답을 처리 중인지 나타내는 플래그

# --- 모델 로드 ---
def load_models():
    """모든 AI 모델을 로드합니다."""
    logger.info("모든 AI 모델을 로드합니다. 시간이 걸릴 수 있습니다...")
    try:
        # Silero VAD
        vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
        (get_speech_timestamps, _, _, VADIterator, _) = utils
        logger.info("1/5 - Silero VAD 모델 로드 완료.")
        
        # Whisper STT
        stt_model = whisper.load_model(STT_MODEL_TYPE)
        logger.info("2/5 - Whisper STT 모델 로드 완료.")

        # Wav2Vec2 SER (Fast)
        feature_extractor_fast = Wav2Vec2FeatureExtractor.from_pretrained(SER_MODEL_FAST)
        model_fast = Wav2Vec2ForSequenceClassification.from_pretrained(SER_MODEL_FAST)
        logger.info("3/5 - 감정 분석(Fast) 모델 로드 완료.")

        # Wav2Vec2 SER (Precise)
        feature_extractor_precise = Wav2Vec2FeatureExtractor.from_pretrained(SER_MODEL_PRECISE)
        model_precise = Wav2Vec2ForSequenceClassification.from_pretrained(SER_MODEL_PRECISE)
        logger.info("4/5 - 감정 분석(Precise) 모델 로드 완료.")

        # Local LLM
        llm = load_llm_model()
        logger.info("5/5 - 로컬 LLM 모델 로드 완료.")
        
        models = {
            "vad": (vad_model, VADIterator),
            "stt": stt_model,
            "ser_fast": (feature_extractor_fast, model_fast),
            "ser_precise": (feature_extractor_precise, model_precise),
            "llm": llm
        }
        logger.info("--- 모든 모델 로드 완료 ---")
        return models
    except Exception as e:
        logger.error(f"모델 로드 중 심각한 오류 발생: {e}")
        exit()

# --- 스레드 함수들 ---

def recording_thread_func(vad_model, vad_iterator):
    """마이크 입력을 지속적으로 감지하고, 음성 구간을 audio_queue에 넣습니다."""
    logger.info("\n>>> AI VTuber가 당신의 말을 듣고 있습니다... <<<")
    
    current_speech_buffer = []
    is_speech_active = False
    last_speech_time = time.time()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=CHUNK_SIZE) as stream:
        while True:
            if is_processing.is_set(): # AI가 말하는 중에는 듣지 않음
                time.sleep(0.1)
                continue

            audio_chunk, _ = stream.read(CHUNK_SIZE)
            audio_chunk_np = audio_chunk.flatten()
            
            vad_result = vad_iterator(audio_chunk_np, return_seconds=True);

            if vad_result and 'start' in vad_result:
                if not is_speech_active:
                    logger.info("음성 시작 감지...")
                    is_speech_active = True
                    current_speech_buffer = [] # 새 음성 시작 시 버퍼 초기화
                current_speech_buffer.append(audio_chunk)
                last_speech_time = time.time()
            
            elif is_speech_active:
                current_speech_buffer.append(audio_chunk)
                # 음성이 잠시 멈췄거나, VAD가 end를 감지했을 때
                if (vad_result and 'end' in vad_result) or (time.time() - last_speech_time > SILENCE_THRESHOLD_SECONDS):
                    is_speech_active = False
                    
                    full_audio_data = np.concatenate(current_speech_buffer, axis=0)
                    duration = len(full_audio_data) / SAMPLE_RATE

                    if duration >= MIN_RECORDING_SECONDS:
                        logger.info(f"음성 구간 종료 ({duration:.2f}초). 처리 큐에 추가합니다.")
                        audio_queue.put(full_audio_data)
                    else:
                        logger.info(f"녹음된 음성이 너무 짧아 무시합니다 ({duration:.2f}초).")
                    current_speech_buffer = []


def audio_processing_thread_func(models):
    """
    audio_queue에서 음성 데이터를 가져와 STT, 감정 분석, LLM 응답 생성, TTS 재생까지 모두 처리합니다.
    """
    global chat_history
    
    while True:
        try:
            # 1. STT (Speech-to-Text)
            recorded_audio = audio_queue.get()
            if recorded_audio is None: break
            
            is_processing.set() # 처리 시작 플래그 설정
            logger.info("\n--- AI 처리 시작 ---")

            # 임시 파일로 저장
            temp_wav_file = f"temp_input_{uuid.uuid4()}.wav"
            write(temp_wav_file, SAMPLE_RATE, np.int16(recorded_audio * 32767))

            result = models["stt"].transcribe(temp_wav_file, language=STT_LANGUAGE, fp16=False)
            user_text = result['text']
            logger.info(f"[사용자] {user_text}")

            if not user_text.strip():
                logger.info("인식된 텍스트가 없어 처리를 중단합니다.")
                os.remove(temp_wav_file)
                is_processing.clear()
                continue

            # 2. SER (Speech Emotion Recognition)
            speech_for_ser, _ = librosa.load(temp_wav_file, sr=SAMPLE_RATE)
            
            # 1단계: 빠른 분석
            feature_extractor_fast, model_fast = models["ser_fast"]
            inputs = feature_extractor_fast(speech_for_ser, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = model_fast(**inputs).logits
            scores = torch.nn.functional.softmax(logits, dim=-1)[0]
            
            # 2단계: 정밀 분석 (필요 시)
            if torch.max(scores).item() < CONFIDENCE_THRESHOLD:
                logger.info("1단계 확신도 낮음. 정밀 분석 수행...")
                feature_extractor_precise, model_precise = models["ser_precise"]
                inputs = feature_extractor_precise(speech_for_ser, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
                with torch.no_grad():
                    logits = model_precise(**inputs).logits
                scores = torch.nn.functional.softmax(logits, dim=-1)[0]
                emotion_labels = model_precise.config.id2label
            else:
                emotion_labels = model_fast.config.id2label

            predicted_emotion = emotion_labels[torch.argmax(scores).item()]
            logger.info(f"[감정 분석] 예측 감정: {predicted_emotion}")

            # 3. LLM 응답 생성
            response_text = generate_response_text(user_text, predicted_emotion, chat_history)
            logger.info(f"[AI 응답] {response_text}")

            # 4. TTS (Text-to-Speech)
            text_to_speech(response_text) # TTS는 내부적으로 스레드를 사용해 비동기 재생

            # 5. 대화 기록 업데이트
            chat_history.append({"user": user_text, "ai": response_text})
            if len(chat_history) > MEMORY_DEPTH:
                chat_history.pop(0) # 가장 오래된 기록 삭제

            os.remove(temp_wav_file)
            logger.info("--- AI 처리 완료 ---\n>>> 다시 당신의 말을 듣고 있습니다... <<<")
            
            # 모든 처리가 끝나고 TTS 재생이 끝날 시간을 약간 기다린 후 플래그 해제
            # (TTS 길이에 따라 조절 가능)
            time.sleep(1) 
            is_processing.clear()

        except Exception as e:
            logger.error(f"오디오 처리 스레드에서 오류 발생: {e}")
            is_processing.clear() # 오류 발생 시에도 플래그 해제


def main():
    """메인 함수: 모델을 로드하고 스레드를 시작합니다."""
    models = load_models()
    
    vad_model, vad_iterator_cls = models["vad"]
    vad_iterator = vad_iterator_cls(vad_model, threshold=VAD_THRESHOLD)

    # 스레드 시작
    recorder = threading.Thread(target=recording_thread_func, args=(vad_model, vad_iterator), daemon=True)
    processor = threading.Thread(target=audio_processing_thread_func, args=(models,), daemon=True)
    
    recorder.start()
    processor.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("프로그램을 종료합니다.")
        audio_queue.put(None) # 스레드 종료 신호
        processor.join()


if __name__ == "__main__":
    main()
