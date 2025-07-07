import logging
import sys

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',    # 파랑
        'INFO': '\033[92m',     # 초록
        'WARNING': '\033[93m',  # 노랑
        'ERROR': '\033[91m',    # 빨강
        'CRITICAL': '\033[95m', # 보라
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"

# 콘솔 핸들러
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_format = ColoredFormatter('[%(asctime)s][%(levelname)s] %(message)s', datefmt='%H:%M:%S')
console_handler.setFormatter(console_format)

# 파일 핸들러
file_handler = logging.FileHandler('app.log', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_format = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(file_format)

# 루트 로거 설정
logger = logging.getLogger('AI_VTuber')
logger.setLevel(logging.DEBUG)
logger.addHandler(console_handler)
logger.addHandler(file_handler)
logger.propagate = False 