import argparse
import os
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


SYSTEM_PROMPT = "You are an API routing agent that returns strict JSON configurations."

class TextFileDataset(Dataset):
    def __init__(self, text, tokenizer, block_size=256):
        self.examples = []
        self.attention_masks = []
        self.labels = []
        
        # Only treat "Prompt:" at the beginning of a line as a sample boundary.
        # Some user prompts legitimately contain the word "Prompt:" themselves.
        raw_samples = re.split(r"(?m)^Prompt:\s*", text)
        
        for s in raw_samples:
            if not s.strip():
                continue
                
            parts = re.split(r"(?m)^Response:\s*", s, maxsplit=1)
            if len(parts) != 2:
                continue
                
            user_prompt = parts[0].strip()
            response_json = parts[1].strip()
            
            # Format using Llama 3.2's official chat dictionary array structure
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": response_json}
            ]
            
            # The chat template automatically appends correct Llama 3.2 token boundaries (<|eot_id|>)
            formatted_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            prompt_text = tokenizer.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True
            )
            
            tokenized = tokenizer(
                formatted_text,
                max_length=block_size,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = tokenized["input_ids"].squeeze(0)
            attention_mask = tokenized["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            prompt_length = min(
                len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"]),
                block_size,
            )
            labels[:prompt_length] = -100
            labels[attention_mask == 0] = -100

            self.examples.append(input_ids)
            self.attention_masks.append(attention_mask)
            self.labels.append(labels)
            
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        return {
            "input_ids": self.examples[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx]
        }


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def fine_tune_model(
    train_file,
    output_dir,
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    epochs=4,
    learning_rate=2e-4, # LoRA thrives on slightly higher learning rates
    block_size=256,
    batch_size=2,
):
    train_file = Path(train_file)
    output_dir = Path(output_dir)
    text = train_file.read_text(encoding="utf-8")

    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    device = get_device()
    dtype = torch.float16 if device.type in {"mps", "cuda"} else torch.float32

    config = AutoConfig.from_pretrained(model_name, token=token)
    # Transformers releases before Llama 3.2 used the key "type", while the
    # newer model config uses "rope_type". Supplying both keeps either version
    # compatible and avoids KeyError: 'type' during model construction.
    if isinstance(config.rope_scaling, dict):
        if "rope_type" in config.rope_scaling and "type" not in config.rope_scaling:
            config.rope_scaling["type"] = config.rope_scaling["rope_type"]
        elif "type" in config.rope_scaling and "rope_type" not in config.rope_scaling:
            config.rope_scaling["rope_type"] = config.rope_scaling["type"]

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=dtype,
        token=token,
    )
    model.to(device)
    model.config.use_cache = False

    # Set up Parameter-Efficient Fine-Tuning (LoRA)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dataset = TextFileDataset(text=text, tokenizer=tokenizer, block_size=block_size)
    if not dataset:
        raise ValueError(
            f"No training examples found in {train_file}. "
            "Expected samples with line-starting 'Prompt:' and 'Response:' fields."
        )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )

    print(f"Device: {device}; dtype: {dtype}")
    print(f"Number of training examples: {len(dataset)}")
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, loss = {total_loss / len(dataloader):.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    # Save adaptation layers and tokenizer configs safely
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuned adaptation layers saved to: {output_dir}")

    return model, tokenizer

def interactive_session(model, tokenizer):
    model.config.use_cache = True
    model.eval()
    print("\n=== Llama 3.2 Continuous Inference API Session ===")
    print("Type 'exit' to quit.\n")
    
    while True:
        try:
            user_input = input("User Prompt: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input:
                continue

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ]
            # Formulate the prompt sequence up to the point where the assistant takes over
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False, # Enforce deterministic output structure
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            generated_tokens = outputs[0, inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            print(f"\nAPI Response:\n{response}")
            print("-" * 50 + "\n")
            
        except KeyboardInterrupt:
            break

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3.2 1B using PEFT/LoRA.")
    parser.add_argument("--train-file", required=True, help="Path to local training text file.")
    parser.add_argument("--output-dir", default="llama_3_2_tuned_agent", help="Directory to save the weights.")
    parser.add_argument("--epochs", type=int, default=4, help="Number of fine-tuning epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size; use 1 if MPS runs out of memory.")
    parser.add_argument("--block-size", type=int, default=256, help="Maximum tokens per training example.")
    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-1B-Instruct", help="Hugging Face model name or local path.")
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.block_size < 1:
        parser.error("--block-size must be at least 1")

    model, tokenizer = fine_tune_model(
        train_file=args.train_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        block_size=args.block_size,
        model_name=args.model_name,
    )

    interactive_session(model, tokenizer)

if __name__ == "__main__":
    main()
