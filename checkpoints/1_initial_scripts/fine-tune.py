# Python Basic Libraries
import argparse
from pathlib import Path
import re

# PyTorch and Transformers
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Dataset for Small Language Model Fine-Tuning
class TextFileDataset(Dataset):
    def __init__(
        self,
        text: str,
        tokenizer,
        max_length: int = 128,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Parse Question/Answer pairs.
        pattern = re.compile(
            r"Question:\s*(.*?)\s*Answer:\s*(.*?)(?=\n\s*Question:|\Z)",
            flags=re.DOTALL,
        )

        self.examples = []

        for question, answer in pattern.findall(text):
            question = question.strip()
            answer = answer.strip()

            prompt = f"Question: {question}\nAnswer: "
            full_text = f"{prompt}{answer}{tokenizer.eos_token}"

            encoded = tokenizer(
                full_text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
            labels = input_ids.clone()

            # Train the model to produce the answer, not to memorize the prompt.
            prompt_length = len(
                tokenizer(
                    prompt,
                    max_length=max_length,
                    truncation=True,
                    add_special_tokens=False,
                )["input_ids"]
            )
            labels[:prompt_length] = -100
            labels[attention_mask == 0] = -100

            if torch.all(labels == -100):
                raise ValueError(
                    "A Question/Answer pair is too long for max_length="
                    f"{max_length}; no answer tokens remain after truncation."
                )

            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

        if not self.examples:
            raise ValueError(
                "No Question/Answer pairs were found. "
                "Check that the file uses 'Question:' and 'Answer:' labels."
            )


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# Change me
def get_data(dataset_location):
    if not dataset_location.exists():
        raise ValueError("Dataset file not found.")
    with open(dataset_location, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def fine_tune_model(
    text,
    epochs,
    model_name="distilgpt2",
    learning_rate=5e-5,
    batch_size=1
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    dataset = TextFileDataset(
        text=text,
        tokenizer=tokenizer
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    device = "cpu"
    model.to(device)
    model.train()

    print(f"Using device: {device}")
    print(f"Number of training examples: {len(dataset)}")

    avg_loss = None
    for epoch in range(epochs):
        total_loss = 0.0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, loss = {avg_loss:.4f}")

    return model, tokenizer

# Change me
def save_model(model, tokenizer, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Fine-tuned model saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fine tune a Small Language Model (distilgpt2) and save it to the mounted model volume."
        )
    )

    parser.add_argument(
        "--dataset_location",
        required=True,
        help="Path to the training file.",
    )

    parser.add_argument(
        "--model_storage_location",
        required=True,
        help=(
            "Path to the directory where the fine-tuned model should be saved.",
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of fine-tuning epochs.",
    )

    args = parser.parse_args()

    train_data = get_data(Path(args.dataset_location))

    model, tokenizer = fine_tune_model(
        text=train_data,
        epochs=args.epochs,
    )

    save_model(model, tokenizer, Path(args.model_storage_location))


if __name__ == "__main__":
    main()
