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
        1. "restyle": Apply a global style change (e.g., "Japanese style", "Modern style").
           * NOTE: Always do "restyle" FIRST if requested.
        2. "replace": Replace specific objects (e.g., "change sofa to leather sofa").

        Output Format: JSON ONLY. Must contain a "steps" list.

        Example 1 (Complex): "Make the room Cyberpunk style, and change the chair to a gaming chair."
        Output:
        {
            "steps": [
                {"action": "restyle", "style_description": "Cyberpunk style, neon lights, futuristic"},
                {"action": "replace", "target_object": "chair", "new_object_desc": "gaming chair"}
            ]
        }

        Example 2 (Simple): "Just change the table to a glass one."
        Output:
        {
            "steps": [
                {"action": "replace", "target_object": "table", "new_object_desc": "glass table"}
            ]
        }

        Example 3 (Chat): "Hello"
        Output:
        {
            "steps": [],
            "reply": "Hello! How can I help with your design?"
        }

        Important: Output valid JSON only. No explanations.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True) # Qwen3 0.6b thinking mode
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
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        # thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        response = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        
        try:
            clean_json = response.strip().replace("```json", "").replace("```", "")
            return json.loads(clean_json)
        except json.JSONDecodeError:
            print(f"❌ JSON 解析失败: {response}")
            return {"steps": [], "reply": "解析指令失败，请重试。"}