import os
from google.cloud import texttospeech
from pydub import AudioSegment
from utils.logger_config import logger
from utils.config_manager import config_manager
from utils.file_manager import save_temp_audio_file

# --- Google Cloud TTS 설정 ---
try:
    # Docker 환경에서는 환경 변수에서 직접 자격 증명을 찾도록 할 수 있습니다.
    # 하지만 로컬 테스트를 위해 파일 경로도 계속 지원합니다.
    key_path = config_manager.get('api_keys', 'google_application_credentials')
    if os.path.exists(key_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
        logger.info(f"Google Cloud credentials set from file: {key_path}")
    
    tts_client = texttospeech.TextToSpeechClient()
    logger.info("Google Cloud TTS client initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing Google Cloud TTS client: {e}")
    logger.warning("TTS functionality will be disabled.")
    tts_client = None

def text_to_speech_sync(text: str) -> str:
    """
    텍스트를 음성으로 변환하고, 생성된 오디오 파일의 경로를 반환합니다. (동기 방식)
    """
    if not tts_client:
        logger.error("TTS client is not initialized. Cannot generate speech.")
        return None

    try:
        logger.info("Synthesizing speech with Google Cloud TTS...")
        
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        voice = texttospeech.VoiceSelectionParams(
            language_code="ko-KR",
            name="ko-KR-Wavenet-A",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
        )
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        
        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        
        # 오디오 데이터를 임시 파일로 저장
        file_path = save_temp_audio_file(response.audio_content, suffix=".mp3")
        logger.info(f"AI response audio saved to: {file_path}")
        return file_path

    except Exception as e:
        logger.error(f"Error during Google Cloud TTS processing: {e}")
        return None

# --- FFmpeg 경로 설정 (기존 코드 유지) ---
# Docker 이미지에는 FFmpeg가 이미 설치되어 있으므로, 이 코드는 주로
# Docker 없이 로컬에서 실행할 때를 위해 유지합니다.
try:
    ffmpeg_path = config_manager.get('paths', 'ffmpeg_path', fallback='ffmpeg')
    if ffmpeg_path.lower() != 'ffmpeg' and not os.path.exists(ffmpeg_path):
        logger.warning(f"FFmpeg path not found: {ffmpeg_path}. Please check config.ini.")
    else:
        AudioSegment.converter = ffmpeg_path
        logger.info(f"FFmpeg path set to: {ffmpeg_path}")
except Exception as e:
    logger.warning(f"Could not read FFmpeg path from config.ini: {e}")



