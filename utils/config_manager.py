import configparser
import os

class ConfigManager:
    def __init__(self, config_path='config.ini'):
        self.config = configparser.ConfigParser()
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"설정 파일이 존재하지 않습니다: {config_path}")
        self.config.read(config_path, encoding='utf-8')

    def get(self, section, key, fallback=None):
        try:
            return self.config.get(section, key, fallback=fallback)
        except Exception as e:
            raise KeyError(f"설정에서 {section}.{key} 값을 불러올 수 없습니다: {e}")

    def getint(self, section, key, fallback=None):
        try:
            return self.config.getint(section, key, fallback=fallback)
        except Exception as e:
            raise KeyError(f"설정에서 {section}.{key} (int) 값을 불러올 수 없습니다: {e}")

    def getfloat(self, section, key, fallback=None):
        try:
            return self.config.getfloat(section, key, fallback=fallback)
        except Exception as e:
            raise KeyError(f"설정에서 {section}.{key} (float) 값을 불러올 수 없습니다: {e}")

config_manager = ConfigManager() 