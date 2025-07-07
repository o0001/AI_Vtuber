from llama_cpp import Llama
from utils.logger_config import logger
from utils.config_manager import config_manager
from typing import List, Dict

# 로컬 LLM 모델 경로 및 n_gpu_layers 설정
LOCAL_LLM_MODEL_PATH = config_manager.get('local_llm', 'model_path')
N_GPU_LAYERS = config_manager.getint('local_llm', 'n_gpu_layers', fallback=0)

# LLM 매개변수 설정
LLM_MAX_TOKENS = config_manager.getint('llm_params', 'max_tokens')
LLM_TEMPERATURE = config_manager.getfloat('llm_params', 'temperature')

llm_model = None

def load_llm_model():
    """LLM 모델을 로드합니다. 이미 로드된 경우, 기존 모델을 반환합니다."""
    global llm_model
    if llm_model is None:
        logger.info("로컬 LLM 모델 로드 중... (시간이 오래 걸릴 수 있습니다)")
        try:
            llm_model = Llama(model_path=LOCAL_LLM_MODEL_PATH, n_gpu_layers=-1, n_ctx=2048, n_batch=512, verbose=True, offload_kqv=True)
            logger.info("로컬 LLM 모델 로드 완료.")
        except Exception as e:
            logger.error(f"로컬 LLM 모델 로드 중 오류 발생: {e}")
            logger.error("GPU 설정(n_gpu_layers)을 확인하거나, 모델 파일 경로를 확인해주세요.")
            exit()
    return llm_model

def generate_response_text(user_text: str, user_emotion: str, chat_history: List[Dict[str, str]]) -> str:
    """로컬 LLM을 사용하여 사용자의 말, 감정, 그리고 이전 대화 기록에 대한 응답을 생성합니다."""
    model = load_llm_model()
    logger.info("로컬 LLM으로 응답 생성 중...")

    # --- 프롬프트 구성 ---
    # 대화 기록을 프롬프트에 포함시켜 맥락을 이해하도록 합니다.
    history_str = "\n".join([f"User: {item['user']}\nYou: {item['ai']}" for item in chat_history])

    prompt = f"""### Instruction:
You are a friendly and empathetic AI VTuber. Your goal is to have a natural and supportive conversation with the user.
Based on the user's input, their emotion, and the recent conversation history, provide a good, short, and natural response in Korean.
Keep your answers concise and conversational.

### Conversation History:
{history_str}

### Current Input:
The user just said: "{user_text}"
They seem to be feeling: "{user_emotion}"

### Response:
"""
    
    try:
        output = model(prompt,
                       max_tokens=LLM_MAX_TOKENS,
                       stop=["###", "\nUser:", "\nYou:"],
                       echo=False,
                       temperature=LLM_TEMPERATURE)
        
        response_text = output["choices"][0]["text"].strip()
        logger.info(f"AI 응답 (텍스트): {response_text}")
        return response_text
    except Exception as e:
        logger.error(f"로컬 LLM 응답 생성 중 오류 발생: {e}")
        return "죄송해요, 지금은 답변을 생각하기가 어렵네요."

