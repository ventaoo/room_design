import torch
import gc
import matplotlib.pyplot as plt
from PIL import Image
from diffusers.utils import load_image

def flush_gpu():
    """清理显存和内存"""
    gc.collect()
    torch.cuda.empty_cache()
    # print("🧹 GPU Cache Flushed")

def load_image_with_aspect_ratio(img_url, max_size=512):
    """加载并调整图片大小，保持长宽比"""
    if isinstance(img_url, str):
        image = load_image(img_url)
    else:
        image = img_url
        
    original_width, original_height = image.size
    ratio = min(max_size / original_width, max_size / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_image

def visualize_change(img_before, img_after, step_num, description):
    """可视化对比图"""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_before)
    plt.title(f"Step {step_num} Before", fontsize=12)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img_after)
    plt.title(f"Step {step_num} After: {description}", fontsize=12)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def apply_mask_visual(image, mask):
    """在图片上叠加 Mask 便于观察"""
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (255, 0, 0, 100))
    final_overlay = Image.composite(overlay, Image.new("RGBA", image.size, (0,0,0,0)), mask)
    return Image.alpha_composite(image, final_overlay)