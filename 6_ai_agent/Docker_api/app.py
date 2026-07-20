import argparse
import os
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from peft import PeftConfig, PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="Classroom AI Agent API")

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def fix_llama_rope_config(config):
    """Support both old Transformers and newer Llama rope config keys."""
    if isinstance(config.rope_scaling, dict):
        if "rope_type" in config.rope_scaling and "type" not in config.rope_scaling:
            config.rope_scaling["type"] = config.rope_scaling["rope_type"]
        elif "type" in config.rope_scaling and "rope_type" not in config.rope_scaling:
            config.rope_scaling["rope_type"] = config.rope_scaling["type"]
    return config

def load_model(adapter_path):
    adapter_path = Path(adapter_path).expanduser()
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"Model adapter directory does not exist: {adapter_path}")
    if not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"No adapter_config.json found in {adapter_path}. "
            "Pass the LoRA output directory created by finetune_ai_agent_new.py."
        )

    token = os.environ.get("HF_TOKEN")
    peft_config = PeftConfig.from_pretrained(adapter_path)
    base_model_name = peft_config.base_model_name_or_path
    device = get_device()
    dtype = torch.float16 if device.type in {"mps", "cuda"} else torch.float32

    print(f"Loading model from {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, token=token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading tokenizer from {adapter_path}")

    config = AutoConfig.from_pretrained(base_model_name, token=token)
    config = fix_llama_rope_config(config)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        config=config,
        torch_dtype=dtype,
        token=token,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.to(device)
    model.eval()

    print("API is ready")

    return model, tokenizer

SYSTEM_PROMPT = "You are an API routing agent that returns strict JSON configurations."

# Load the adapter from the Kubernetes model volume by default. MODEL_PATH keeps
# local development and differently named model directories configurable.
# model, tokenizer = load_model(os.environ.get("MODEL_PATH", "/models/ai_agent_model"))
model, tokenizer = load_model("../model_training/models/llama_3_2_tuned_agent")

# Define the structure you want the students to receive
class ParsedResponse(BaseModel):
    intent: str
    entities: list[str]
    confidence: float

@app.post("/parse")
def parse_prompt(prompt: str):

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return response

@app.get("/healthz")
def healthz():
    return {"status": "ok"}