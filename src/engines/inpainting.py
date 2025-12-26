import torch
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
from src.config import cfg

class InpaintingEngine:
    def __init__(self):
        self.device = cfg['device'] if torch.cuda.is_available() else "cpu"
        print(f"🎨 初始化重绘引擎 ({self.device})...")

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            cfg['models']['inpainting'],
            torch_dtype=torch.float16,
            safety_checker=None
        )
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(self.device)
        self.pipe.enable_model_cpu_offload()

    def replace_item(self, image, mask, prompt, negative_prompt="", seed=-1, strength=None):
        if strength is None:
            strength = cfg['parameters']['inpainting_strength']

        generator = torch.Generator(device=self.device).manual_seed(seed) if seed != -1 else None
        
        enhanced_prompt = f"{prompt}, (high quality, photorealistic, 8k, interior design:1.2)"
        enhanced_neg = f"{negative_prompt}, bad anatomy, blurry, text, watermark, low quality"

        result = self.pipe(
            prompt=enhanced_prompt,
            negative_prompt=enhanced_neg,
            image=image,
            mask_image=mask,
            num_inference_steps=30,
            strength=strength,
            guidance_scale=cfg['parameters']['guidance_scale'],
            generator=generator,
            width=(image.size[0] // 8) * 8,
            height=(image.size[1] // 8) * 8
        ).images[0]

        return result