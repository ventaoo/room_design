import os
import torch
import urllib.request
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from segment_anything import sam_model_registry, SamPredictor
from src.config import cfg

class VisionEngine:
    def __init__(self):
        self.device = cfg['device'] if torch.cuda.is_available() else "cpu"
        print(f"👁️ 初始化视觉引擎 ({self.device})...")

        # Load DINO
        self.dino_processor = AutoProcessor.from_pretrained(cfg['models']['grounding_dino'])
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            cfg['models']['grounding_dino']
        ).to(self.device)

        # Download & Load SAM
        self._check_download_sam()
        self.sam = sam_model_registry[cfg['models']['sam_type']](checkpoint=cfg['models']['sam_checkpoint'])
        self.sam.to(device=self.device)
        self.sam_predictor = SamPredictor(self.sam)

    def _check_download_sam(self):
        ckpt_path = cfg['models']['sam_checkpoint']
        if not os.path.exists(ckpt_path):
            print(f"⬇️ 下载 SAM 权重...")
            urllib.request.urlretrieve(cfg['models']['sam_url'], ckpt_path)
            print("✅ 下载完成")

    def detect_object(self, image, text_prompt, box_threshold=0.3, text_threshold=0.25):
        if not text_prompt.endswith("."): text_prompt += "."
        
        inputs = self.dino_processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.dino_model(**inputs)

        w, h = image.size
        target_sizes = torch.tensor([[h, w]], device=self.device)
        results = self.dino_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=box_threshold, text_threshold=text_threshold, target_sizes=target_sizes
        )[0]
        return results["boxes"], results["labels"]

    def get_mask(self, image, text_prompt):
        print(f"🔍 正在寻找: '{text_prompt}'...")
        boxes, _ = self.detect_object(image, text_prompt)
        
        if len(boxes) == 0:
            print("⚠️ 未找到目标物体。")
            return Image.new("L", image.size, 0)

        image_np = np.array(image)
        self.sam_predictor.set_image(image_np)
        
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            boxes, image_np.shape[:2]
        ).to(self.device)
        
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None, point_labels=None,
            boxes=transformed_boxes, multimask_output=False
        )
        
        final_mask_tensor = torch.any(masks, dim=0).squeeze()
        final_mask_np = final_mask_tensor.cpu().numpy().astype(np.uint8) * 255
        return Image.fromarray(final_mask_np, mode="L")