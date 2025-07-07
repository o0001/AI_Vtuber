# LLM 서비스 모듈 (샘플)
# 실제 LLM 모델 연동 시 이 파일을 수정하세요.

def load_llm_model():
    # 실제로는 LLM 모델을 로드해야 함
    # 여기서는 더미 객체 반환
    return None

def generate_response_text(user_text, emotion, chat_history):
    # 실제로는 LLM을 이용해 답변을 생성해야 함
    # 여기서는 간단한 규칙 기반 답변
    response = f"[감정: {emotion}] {user_text}에 대한 답변입니다."
    return response 