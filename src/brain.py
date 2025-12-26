import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.config import cfg

class DesignBrain:
    def __init__(self):
        self.device = cfg['device'] if torch.cuda.is_available() else "cpu"
        print(f"🧠 初始化规划大脑 ({self.device})...")
        
        model_id = cfg['models']['llm']
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def analyze_intent(self, user_text):
        system_prompt = """
        You are a Senior Interior Design Planner. Break down the user's request into a sequential execution plan.
        Supported Actions:
        1. "restyle": Apply a global style change (e.g., "Japanese style"). Always do "restyle" FIRST if requested.
        2. "replace": Replace specific objects (e.g., "change sofa to leather sofa").
        
        Output Format: JSON ONLY. Must contain a "steps" list.
        Example: {"steps": [{"action": "restyle", "style_description": "Modern"}, {"action": "replace", "target_object": "chair", "new_object_desc": "red chair"}]}
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        try:
            clean_json = response.strip().replace("```json", "").replace("```", "")
            return json.loads(clean_json)
        except json.JSONDecodeError:
            print(f"❌ JSON 解析失败: {response}")
            return {"steps": [], "reply": "解析指令失败，请重试。"}