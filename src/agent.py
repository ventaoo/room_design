from src.brain import DesignBrain
from src.utils import flush_gpu, visualize_change

class DesignAgent:
    def __init__(self):
        self.brain = DesignBrain()

    def run(self, original_image, user_text):
        print(f"\n📨 收到用户指令: '{user_text}'")
        
        # 1. 规划
        plan = self.brain.analyze_intent(user_text)
        steps = plan.get("steps", [])
        
        if not steps:
            return plan.get("reply", "未识别到任务。")
            
        print(f"📋 方案生成: {len(steps)} 个步骤")

        current_image = original_image.copy()
        
        # 2. 依次执行
        for i, step in enumerate(steps):
            action = step["action"]
            desc = step.get('style_description') or f"{step.get('target_object')} -> {step.get('new_object_desc')}"
            print(f"\n▶️ [Step {i+1}] 执行: {desc}")
            
            image_before = current_image.copy()
            
            if action == "restyle":
                current_image = self._execute_restyle(current_image, step["style_description"])
            elif action == "replace":
                current_image = self._execute_replace(current_image, step["target_object"], step["new_object_desc"])
            
            visualize_change(image_before, current_image, i+1, desc)
            
        print("\n🎉 任务完成！")
        return current_image

    def _execute_restyle(self, image, prompt):
        flush_gpu()
        # 动态导入以节省初始化时间/内存，且确保每次只加载一个大模型
        from src.engines.design import InteriorDesignEngine
        engine = InteriorDesignEngine()
        result, _ = engine.generate_design(image, prompt)
        
        del engine # 立即释放
        flush_gpu()
        return result

    def _execute_replace(self, image, target, new_desc):
        flush_gpu()
        
        # Phase 1: Vision
        from src.engines.vision import VisionEngine
        vision = VisionEngine()
        mask = vision.get_mask(image, target)
        del vision
        flush_gpu()
        
        if mask.getbbox() is None:
            return image
            
        # Phase 2: Inpainting
        from src.engines.inpainting import InpaintingEngine
        inpainter = InpaintingEngine()
        result = inpainter.replace_item(image, mask, new_desc)
        del inpainter
        flush_gpu()
        
        return result