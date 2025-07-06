import uvicorn
import librosa
import torch
import httpx
import os
from fastapi import FastAPI, UploadFile, File, Form
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from utils.config_manager import config_manager
from utils.logger_config import logger

# 서버 설정
FINAL_API_ENDPOINT = config_manager.get('servers', 'final_api_endpoint')

# 모델 설정
MODEL_FAST_NAME = config_manager.get('models', 'ser_model_fast')
MODEL_PRECISE_NAME = config_manager.get('models', 'ser_model_precise')

# 분석 설정
CONFIDENCE_THRESHOLD = config_manager.getfloat('analysis', 'confidence_threshold')

# 오디오 설정
SAMPLE_RATE = config_manager.getint('audio', 'sample_rate')

# 임시 디렉토리 설정
TEMP_ANALYSIS_AUDIO_DIR = config_manager.get('paths', 'temp_analysis_audio_dir')

# --- 감정 분석 모델 로드 ---
logger.info("감정 분석 모델을 로드합니다...")
try:
    logger.info(f"[1/2] 빠른 분석 모델 로드 중: {MODEL_FAST_NAME}")
    feature_extractor_fast = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_FAST_NAME)
    model_fast = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_FAST_NAME)
    emotion_labels_fast = model_fast.config.id2label
    logger.info("빠른 분석 모델 로드 완료.")

    logger.info(f"[2/2] 정밀 분석 모델 로드 중: {MODEL_PRECISE_NAME} (시간이 걸릴 수 있습니다)")
    feature_extractor_precise = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_PRECISE_NAME)
    model_precise = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PRECISE_NAME)
    emotion_labels_precise = model_precise.config.id2label
    logger.info("정밀 분석 모델 로드 완료.")

except Exception as e:
    logger.error(f"감정 분석 모델 로드 중 오류 발생: {e}")
    exit()

app = FastAPI()

async def send_analysis_to_final_api(analysis_data: dict):
    """분석된 최종 결과를 API 서버로 비��기적으로 전송합니다."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(FINAL_API_ENDPOINT, json=analysis_data, timeout=10.0)
            response.raise_for_status()
            logger.info(f"최종 API 서버 응답: {response.status_code}")
        except httpx.ConnectError:
            logger.error(f"오류: 최종 API 서버({FINAL_API_ENDPOINT})에 연결할 수 없습니다.")
        except httpx.RequestError as e:
            logger.error(f"최종 API 요청 중 오류 발생: {e}")

@app.post("/analyze/")
async def analyze_audio_and_text(text: str = Form(...), audio_file: UploadFile = File(...)):
    """오디오와 텍스트를 받아 2단계에 걸쳐 감정과 의도를 분석하는 엔드포인트"""
    logger.info(f"분석 요청 수신: 텍스트 - \"{text}\"")
    
    try:
        async with manage_temp_file(audio_file, TEMP_ANALYSIS_AUDIO_DIR) as temp_filename:
            speech, sr = librosa.load(temp_filename, sr=SAMPLE_RATE)

            # --- 1단계: 빠른 기본 분석 ---
            logger.info("[1단계] 빠른 분석을 수행합니다...")
            input_features_fast = feature_extractor_fast(speech, sampling_rate=sr, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits_fast = model_fast(**input_features_fast).logits
            
            scores_fast = torch.nn.functional.softmax(logits_fast, dim=-1)[0]
            highest_score_fast = torch.max(scores_fast).item()
            
            analysis_level = "fast"
            predicted_emotion = emotion_labels_fast[torch.argmax(scores_fast).item()]
            emotion_scores = {emotion_labels_fast[i]: round(s.item(), 4) for i, s in enumerate(scores_fast)}

            # --- 2단계: 정밀 분석 (필요 시) ---
            if highest_score_fast < CONFIDENCE_THRESHOLD:
                logger.info(f"1단계 확신도({highest_score_fast:.2f})가 낮아, 2단계 정밀 분석을 수행합니다...")
                analysis_level = "precise"
                
                input_features_precise = feature_extractor_precise(speech, sampling_rate=sr, return_tensors="pt", padding=True)
                with torch.no_grad():
                    logits_precise = model_precise(**input_features_precise).logits
                
                scores_precise = torch.nn.functional.softmax(logits_precise, dim=-1)[0]
                predicted_emotion = emotion_labels_precise[torch.argmax(scores_precise).item()]
                emotion_scores = {emotion_labels_precise[i]: round(s.item(), 4) for i, s in enumerate(scores_precise)}
            else:
                logger.info(f"1단계 확신도({highest_score_fast:.2f})가 충분하여 빠른 분석 결과 사용.")

            sentence_type = "질문" if text.endswith("?") else "일반 진술"

            analysis_result = {
                "original_text": text,
                "text_analysis": {"sentence_type": sentence_type},
                "audio_analysis": {
                    "predicted_emotion": predicted_emotion,
                    "emotion_scores": emotion_scores,
                    "analysis_level": analysis_level
                }
            }
            
            logger.info(f"최종 분석 결과: {analysis_result}")

            await send_analysis_to_final_api(analysis_result)
            return {"status": "success", "analysis": analysis_result}

    except Exception as e:
        logger.error(f"분석 중 오류 발생: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    logger.info("2단계 감정 분석 서버를 시작합니다. (http://localhost:9001)")
    uvicorn.run(app, host="0.0.0.0", port=9001)



