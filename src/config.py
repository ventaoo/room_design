import yaml
import os

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        # 兼容从 src 目录或根目录运行的情况
        config_path = os.path.join("..", config_path)
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

# 单例模式加载配置
cfg = load_config()