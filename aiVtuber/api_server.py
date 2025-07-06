from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn
from llm_service import generate_response_text
from audio_service import text_to_speech

from utils.logger_config import logger

# --- 모델 및 시스템 설정 ---
app = FastAPI()

# 데이터 모델 정의
class AnalysisResult(BaseModel):
    original_text: str
    text_analysis: Dict[str, Any]
    audio_analysis: Dict[str, Any]

@app.post("/receive_analysis")
async def receive_analysis_and_respond(result: AnalysisResult):
    """분석 결과를 받아, 응답을 생성하고, 음성으로 말하는 최종 엔드포인트."""
    logger.info("--- 최종 분석 결과 수신 (api_server) ---")
    logger.info(f"  - 원본 텍스트: {result.original_text}")
    logger.info(f"  - 예측 감정: {result.audio_analysis.get('predicted_emotion')}")
    logger.info("----------------------------------------")

    response_text = generate_response_text(
        user_text=result.original_text,
        user_emotion=result.audio_analysis.get('predicted_emotion')
    )

    if response_text:
        text_to_speech(response_text)
    
    return {"status": "success", "response_text": response_text}

if __name__ == "__main__":
    logger.info("AI 응답 생��� 및 표현 서버를 시작합니다. (http://localhost:8000)")
    uvicorn.run(app, host="0.0.0.0", port=8000)