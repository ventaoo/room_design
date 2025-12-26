import os
import matplotlib.pyplot as plt
from src.agent import DesignAgent
from src.utils import load_image_with_aspect_ratio

def main():
    # 1. 初始化 Agent (会加载 LLM)
    agent = DesignAgent()
    
    # 2. 准备输入
    img_url = "https://images.unsplash.com/photo-1513694203232-719a280e022f?q=80&w=1000&auto=format&fit=crop"
    init_image = load_image_with_aspect_ratio(img_url)
    
    prompt = "Change this room to a minimalist style, and change the bed to a Japanese futon with green sheets."
    
    # 3. 运行
    result = agent.run(init_image, prompt)
    
    # 4. 最终展示
    if not isinstance(result, str):
        plt.figure(figsize=(10, 5))
        plt.imshow(result)
        plt.axis("off")
        plt.title("Final Result")
        plt.show()

if __name__ == "__main__":
    token = os.environ.get('HF_TOKEN', None)
    if token:
        os.environ['HF_TOKEN'] = token
    
    main()