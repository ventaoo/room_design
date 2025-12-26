import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import MidasDetector
from src.config import cfg

class InteriorDesignEngine:
    def __init__(self):
        self.device = cfg['device'] if torch.cuda.is_available() else "cpu"
        print(f"🏗️ 初始化设计引擎 ({self.device})...")

        self.preprocessor = MidasDetector.from_pretrained(cfg['models']['depth_preprocessor'])

        controlnet = ControlNetModel.from_pretrained(
            cfg['models']['controlnet_depth'],
            torch_dtype=torch.float16
        )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            cfg['models']['sd_base'],
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None
        )

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(self.device)
        self.pipe.enable_model_cpu_offload()

    def generate_design(self, original_image, prompt, negative_prompt="", seed=-1, strength=None):
        if strength is None:
            strength = cfg['parameters']['controlnet_strength']
            
        depth_image = self.preprocessor(original_image)

        enhanced_prompt = (
            f"{prompt}, interior design, 8k uhd, dslr, soft lighting, high quality, "
            "film grain, Fujifilm XT3"
        )
        enhanced_negative_prompt = (
            f"{negative_prompt}, (deformed iris, deformed pupils, semi-realistic, cgi, 3d, sketch, anime:1.4), "
            "text, close up, cropped, low quality, bad anatomy, bad proportions"
        )

        generator = torch.Generator(device=self.device).manual_seed(seed) if seed != -1 else None

        result = self.pipe(
            prompt=enhanced_prompt,
            negative_prompt=enhanced_negative_prompt,
            image=depth_image,
            num_inference_steps=cfg['parameters']['sd_inference_steps'],
            controlnet_conditioning_scale=strength,
            guidance_scale=cfg['parameters']['guidance_scale'],
            generator=generator
        ).images[0]

        return result, depth_image