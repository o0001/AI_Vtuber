import os
import io
import torch
import librosa
import asyncio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import whisper
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
from pydub import AudioSegment

from utils.config_manager import config_manager
from utils.logger_config import logger
from utils.file_manager import manage_temp_audio_file, get_temp_file_path
from llm_service import generate_response_text, load_llm_model
from audio_service import text_to_speech_sync


app = FastAPI()


MODELS = {}
CHAT_HISTORY = []
api_lock = asyncio.Lock()  
MEMORY_DEPTH = config_manager.getint('llm_params', 'memory_depth', fallback=5)
STT_MODEL_TYPE = config_manager.get('models', 'stt_model_type')
SER_MODEL_NAME = config_manager.get('models', 'ser_model')
STT_LANGUAGE = config_manager.get('stt', 'language')

def load_models():
    """애플리케이션 시작 시 AI 모델을 로드합니다."""
    logger.info("--- AI VTuber 서버 시작 ---")
    logger.info("AI 모델을 로드합니다. 시간이 걸릴 수 있습니다...")
    try:
        # STT (Whisper)
        MODELS["stt"] = whisper.load_model(STT_MODEL_TYPE)
        logger.info(f"1/3 - Whisper STT 모델 ({STT_MODEL_TYPE}) 로드 완료.")

        # SER (Wav2Vec2)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(SER_MODEL_NAME)
        ser_model = Wav2Vec2ForSequenceClassification.from_pretrained(SER_MODEL_NAME)
        MODELS["ser"] = (feature_extractor, ser_model)
        logger.info("2/3 - 감정 분석(SER) 모델 로드 완료.")

        # LLM (Llama.cpp)
        MODELS["llm"] = load_llm_model()
        logger.info("3/3 - 로컬 LLM 모델 로드 완료.")
        
        logger.info("--- 모든 모델 로드 완료. 서버가 준비되었습니다. ---")
    except Exception as e:
        logger.error(f"모델 로드 중 심각한 오류 발생: {e}", exc_info=True)
        raise RuntimeError("Failed to load AI models.") from e

@app.on_event("startup")
def startup_event():
    load_models()

async def convert_audio_to_wav(audio_file: UploadFile) -> bytes:
    """업로드된 오디오 파일을 WAV 형식의 바이트로 변환합니다."""
    try:
        audio_bytes = await audio_file.read()
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav_bytes_io = io.BytesIO()
        audio_segment.export(wav_bytes_io, format="wav")
        return wav_bytes_io.getvalue()
    except Exception as e:
        logger.error(f"오디오 변환 중 오류: {e}")
        raise HTTPException(status_code=400, detail=f"오디오 파일 처리 중 오류 발생: {e}")

def transcribe_audio(temp_wav_path: str) -> str:
    """오디오 파일 경로를 받아 텍스트로 변환합니다."""
    use_fp16 = torch.cuda.is_available()
    result = MODELS["stt"].transcribe(temp_wav_path, language=STT_LANGUAGE, fp16=use_fp16)
    user_text = result['text']
    logger.info(f"[사용자] {user_text}")
    return user_text

def recognize_emotion(temp_wav_path: str) -> str:
    """오디오 파일 경로를 받아 감정을 분석합니다."""
    speech, sr = librosa.load(temp_wav_path, sr=16000)
    feature_extractor, ser_model = MODELS["ser"]
    inputs = feature_extractor(speech, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = ser_model(**inputs).logits
    scores = torch.nn.functional.softmax(logits, dim=-1)[0]
    emotion_labels = ser_model.config.id2label
    predicted_emotion = emotion_labels[torch.argmax(scores).item()]
    logger.info(f"[감정 분석] 예측 감정: {predicted_emotion}")
    return predicted_emotion

def update_chat_history(user_text: str, ai_text: str):
    """대화 기록을 업데이트합니다."""
    CHAT_HISTORY.append({"user": user_text, "ai": ai_text})
    if len(CHAT_HISTORY) > MEMORY_DEPTH:
        CHAT_HISTORY.pop(0)

@app.post("/process-audio/")
async def process_audio_endpoint(audio_file: UploadFile = File(...)):
    """오디오 파일을 받아 전체 AI 파이프라인을 실행합니다."""
    if not MODELS:
        raise HTTPException(status_code=503, detail="AI 모델이 아직 준비되지 않았습니다.")

    async with api_lock:
        try:
            logger.info("\n--- 오디오 처리 요청 수신 ---")
            wav_bytes = await convert_audio_to_wav(audio_file)

            with manage_temp_audio_file(wav_bytes, suffix=".wav") as temp_wav_path:
                
                loop = asyncio.get_running_loop()
                
                user_text = await loop.run_in_executor(
                    None, transcribe_audio, temp_wav_path
                )

                if not user_text.strip():
                    logger.info("인식된 텍스트가 없어 처리를 중단합니다.")
                    return JSONResponse(content={"user_text": "", "ai_text": "", "audio_path": ""})

                predicted_emotion = await loop.run_in_executor(
                    None, recognize_emotion, temp_wav_path
                )
                
                response_text = await loop.run_in_executor(
                    None, generate_response_text, user_text, predicted_emotion, CHAT_HISTORY
                )
                logger.info(f"[AI 응답] {response_text}")

                response_audio_path = await loop.run_in_executor(
                    None, text_to_speech_sync, response_text
                )
                if not response_audio_path:
                    raise HTTPException(status_code=500, detail="TTS 음성 생성에 실패했습니다.")

                update_chat_history(user_text, response_text)
                
                logger.info("--- 오디오 처리 완료 ---")
                
                response_audio_filename = os.path.basename(response_audio_path)
                return JSONResponse(content={
                    "user_text": user_text,
                    "ai_text": response_text,
                    "audio_path": response_audio_filename
                })

        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            logger.error(f"오디오 처리 중 예상치 못한 오류 발생: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.get("/audio/{file_name:path}")
async def get_audio_file(file_name: str):
    """생성된 오디오 파일을 클라이언트가 다운로드할 수 있도록 제공합니다."""
    safe_path = get_temp_file_path(file_name)
    if not os.path.exists(safe_path):
        raise HTTPException(status_code=404, detail=f"Audio file not found: {file_name}")
    return FileResponse(safe_path, media_type="audio/mpeg", filename=os.path.basename(safe_path))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

