import os
import uuid
import threading
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from utils.logger_config import logger

def text_to_speech(text: str):
    """생성된 텍스트를 음성 파일로 변환하고 재생합니다."""
    try:
        logger.info("TTS로 음성 합성 중...")
        tts = gTTS(text=text, lang='ko')
        
        temp_audio_file = f"response_{uuid.uuid4()}.mp3"
        tts.save(temp_audio_file)
        
        logger.info("AI 음성 재생 (백그라운드)...")
        threading.Thread(target=lambda: play_and_cleanup(temp_audio_file)).start()

    except Exception as e:
        logger.error(f"TTS 처리 중 오류 발생: {e}")

def play_and_cleanup(audio_file_path: str):
    try:
        audio = AudioSegment.from_file(audio_file_path)
        play(audio)
    except Exception as e:
        logger.error(f"음성 재생 중 오류 발생: {e}")
    finally:
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)
            logger.info("재생 완료 및 임시 파일 삭제.")
