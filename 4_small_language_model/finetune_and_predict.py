import argparse
from pathlib import Path
import os

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

access_token = os.environ.get("MY_TOKEN")
os.environ["HF_TOKEN"] = str(access_token)

class TextFileDataset(Dataset):
    def __init__(self, text, tokenizer, block_size=64):
        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
        )["input_ids"][0]

        self.examples = []

        for i in range(0, len(tokens) - block_size, block_size):
            chunk = tokens[i : i + block_size]
            self.examples.append(chunk)

        # If the file is very small, still create one padded example
        if len(self.examples) == 0:
            chunk = tokens[:block_size]
            padded = torch.full((block_size,), tokenizer.eos_token_id, dtype=torch.long)
            padded[: len(chunk)] = chunk
            self.examples.append(padded)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids = self.examples[idx]
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        }


def fine_tune_model(
    train_file,
    output_dir,
    model_name="distilgpt2",
    epochs=1,
    learning_rate=5e-5,
    block_size=64,
):
    train_file = Path(train_file)
    output_dir = Path(output_dir)

    text = train_file.read_text(encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # GPT2-style models do not have a default padding token
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    dataset = TextFileDataset(
        text=text,
        tokenizer=tokenizer,
        block_size=block_size,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    print(f"Using device: {device}")
    print(f"Number of training examples: {len(dataset)}")

    for epoch in range(epochs):
        total_loss = 0.0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
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
        default="fine_tuned_distilgpt2",
        help="Directory to save the fine-tuned model.",
    )

    parser.add_argument(
        "--question",
        required=True,
        help="Question to ask the fine-tuned model.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
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