from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.common import project_path


@dataclass
class GenerationConfig:
    max_new_tokens: int
    temperature: float
    top_p: float


class QwenGenerator:
    def __init__(self, model_name: str, cache_dir: str | None = None, load_in_4bit: bool = True):
        kwargs = {}
        if cache_dir:
            kwargs["cache_dir"] = str(project_path(cache_dir))

        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            quantization_config=quantization_config,
            **kwargs,
        )

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        messages = [
            {"role": "system", "content": "You are a careful environmental marketing compliance assistant."},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=config.temperature > 0,
                temperature=config.temperature,
                top_p=config.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
