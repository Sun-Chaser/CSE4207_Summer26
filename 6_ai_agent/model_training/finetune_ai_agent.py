import argparse
from pathlib import Path
import os

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


class TextFileDataset(Dataset):
    def __init__(self, text, tokenizer, block_size=256):
        self.examples = []
        self.attention_masks = []
        
        raw_samples = text.split("Prompt:")
        
        for s in raw_samples:
            if not s.strip():
                continue
                
            # Split the prompt from the response block
            parts = s.split("Response:")
            if len(parts) != 2:
                continue
                
            user_prompt = parts[0].strip()
            response_json = parts[1].strip()
            
            # Structure the text with crystal-clear boundary tokens
            formatted_text = (
                f"<|user|>\n{user_prompt}\n"
                f"<|assistant|>\n{response_json}{tokenizer.eos_token}"
            )
            
            tokenized = tokenizer(
                formatted_text,
                max_length=block_size,
                padding="max_length",  # Keep padding, but now it fills to 256
                truncation=True,       # Will safely keep the entire JSON intact now
                return_tensors="pt"
            )
            
            self.examples.append(tokenized["input_ids"].squeeze(0))
            self.attention_masks.append(tokenized["attention_mask"].squeeze(0))
            
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        return {
            "input_ids": self.examples[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.examples[idx].clone()
        }
        


def fine_tune_model(
    train_file,
    output_dir,
    model_name="distilgpt2",
    epochs=4,               # Changed from 1 to 4 (the sweet spot)
    learning_rate=3e-5,     # Adjusted slightly down to prevent overfitting on 30 examples
    block_size=256,         # Increased block_size to ensure entire Prompt+JSON fits in one sequence
):
    train_file = Path(train_file)
    output_dir = Path(output_dir)

    text = train_file.read_text(encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # GPT2-style models do not have a default padding token
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Force the use of standard cross-entropy loss for causal language modeling
    model.config.loss_type = "ForCausalLMLoss"

    dataset = TextFileDataset(
        text=text,
        tokenizer=tokenizer,
        block_size=block_size,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,       # Keeping batch_size=4 gives ~7-8 updates per epoch with 30 examples
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,  # Added a touch of weight decay to keep the small model generalized
    )

    # --- DATASET DIAGNOSTIC PRINT BLOCK ---
    print("\n" + "="*50)
    print("DIAGNOSTIC: INSPECTING THE PARSED DATASET")
    print("="*50)

    # Print the total number of items found
    print(f"Total samples parsed: {len(dataset)}")

    # Loop through and print the first 3 examples to verify formatting
    sample_limit = min(5, len(dataset))
    for idx in range(sample_limit):
        sample = dataset[idx]
        input_ids = sample["input_ids"]
        
        # 1. Decode back to text to see if our string splitting worked perfectly
        decoded_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        
        print(f"\n--- [Example Sample #{idx + 1}] ---")
        print(">>> RAW STRING FORMAT PASSED TO MODEL:")
        print(decoded_text)
        
        print("\n>>> TENSOR DATA SHAPE:")
        print(f"Input IDs Shape: {input_ids.shape}")
        print(f"Attention Mask Shape: {sample['attention_mask'].shape}")
        
        # Optional: View actual tokens to check padding alignment
        # print(">>> RAW TOKEN IDs (First 20 tokens):", input_ids[:20].tolist())
        print("-" * 40)

    print("="*50 + "\n")
    # --- END DIAGNOSTIC PRINT BLOCK ---

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    print(f"Using device: {device}")
    print(f"Number of training examples: {len(dataset)}")

    for epoch in range(epochs):
        total_loss = 0.0

        # Inside your training loop:
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device) # <-- Add this line
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,                  # <-- Add this line
                labels=labels,
            )

            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, loss = {avg_loss:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Fine-tuned model saved to: {output_dir}")

    return model, tokenizer


def answer_question(
    model,
    tokenizer,
    question,
    max_new_tokens=80,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    prompt = f"Question: {question}\nAnswer:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,  # Essential: halts immediately when EOS token is sampled
        )

    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    # Optional: extract only the answer part
    if "Answer:" in generated_text:
        answer = generated_text.split("Answer:", 1)[1].strip()
    else:
        answer = generated_text.strip()

    # Optional: stop if the model starts generating another question
    if "Question:" in answer:
        answer = answer.split("Question:", 1)[0].strip()

    return answer


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune distilgpt2 on a text file and generate from a prompt."
    )

    parser.add_argument(
        "--train-file",
        required=True,
        help="Path to local training text file.",
    )

    parser.add_argument(
        "--output-dir",
        default="fine_tuned_ai_agent",
        help="Directory to save the fine-tuned model.",
    )

    parser.add_argument(
        "--question",
        # required=True,
        help="Question to ask the fine-tuned model.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of fine-tuning epochs.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Maximum number of new tokens to generate.",
    )

    args = parser.parse_args()

    model, tokenizer = fine_tune_model(
        train_file=args.train_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
    )

    if args.question:
        answer = answer_question(
            model=model,
            tokenizer=tokenizer,
            question=args.question,
            max_new_tokens=args.max_new_tokens,
        )

        print("\nQuestion:")
        print(args.question)

        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    main()