# Python Basic Libraries
import argparse
import os
from pathlib import Path

# Amazon S3
import boto3
from botocore.exceptions import ClientError

# SQL Database
import psycopg2
from psycopg2.extras import Json

# PyTorch and Transformers
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# AWS S3 Configuration
BUCKET_NAME = os.environ.get("BUCKET_NAME", "cse4207-lab-rebuild")
MODEL_VOLUME_PATH = Path(os.environ.get("MODEL_VOLUME_PATH", "/models"))
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-2")

s3 = boto3.resource(
    "s3",
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    region_name=AWS_REGION
)

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

def normalize_s3_key(file_key):
    """
    Convert input into a clean S3 object key.
    Not necessary if students know their file path

    Accepts:
    - train.csv
    - data/train.csv
    - s3://BUCKET_NAME/data/train.csv
    """
    file_key = str(file_key)

    if file_key.startswith("s3://"):
        path_without_prefix = file_key.replace("s3://", "", 1)
        parts = path_without_prefix.split("/", 1)

        if len(parts) != 2:
            raise ValueError(f"Invalid S3 path: {file_key}")

        bucket_name, object_key = parts

        if bucket_name != BUCKET_NAME:
            raise ValueError(
                f"Expected bucket '{BUCKET_NAME}', but got bucket '{bucket_name}'."
            )

        return object_key
    
    # Remove leading slashes
    return file_key.lstrip("/")

# Cloud reading from S3 Bucket
def read_from_s3(file_key):
    object_key = normalize_s3_key(file_key)

    obj = s3.Object(BUCKET_NAME, object_key)

    try:
        body = obj.get()["Body"].read()
    except ClientError as error:
        error_code = error.response.get("Error", {}).get("Code")
        if error_code in {"404", "NoSuchKey", "NotFound"}:
            raise FileNotFoundError(
                f"No S3 object found for dataset '{file_key}'. Checked S3 path: "
                f"s3://{BUCKET_NAME}/{object_key}"
            ) from error
        raise

    text = body.decode("utf-8")
    return text


def get_data(dataset_location):
    return read_from_s3(dataset_location)

# psycopg2 Database Connection and Logging
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "slm-service"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "slmDB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        connect_timeout=5,
    )

# Logging training events to Postgres
def log_training_event(job_type, status, payload):
    try:
        with get_db_connection() as connection:
            with connection.cursor() as cursor:
                # Create table if not exists
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS training_logs (
                        id SERIAL PRIMARY KEY,
                        job_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        payload JSONB,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                # Log information
                cursor.execute(
                    """
                    INSERT INTO training_logs (job_type, status, payload)
                    VALUES (%s, %s, %s)
                    """,
                    (job_type, status, Json(payload)),
                )
    except psycopg2.Error as error:
        print(f"Warning: Failed to log metrics to Postgres: {error}")

# Do not change me
def fine_tune_model(
    text,
    model_name="distilgpt2",
    epochs=1,
    learning_rate=5e-5,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.loss_type = "ForCausalLMLoss"

    dataset = TextFileDataset(
        text=text,
        tokenizer=tokenizer,
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

        log_training_event(
            job_type="fine-tune",
            status="running",
            payload={"loss": round(avg_loss, 4), "epoch": epoch + 1},
        )

    log_training_event(
        job_type="fine-tune",
        status="completed",
        payload={
            "loss": round(avg_loss, 4) if avg_loss is not None else None,
            "epochs": epochs,
        },
    )

    return model, tokenizer

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
        help="S3 key or S3 path to the training file.",
    )

    parser.add_argument(
        "--model_storage_location",
        required=True,
        help=(
            "Local file path on the mounted persistent volume where the model should "
            "be saved. Relative paths are saved under /models."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of fine-tuning epochs.",
    )

    args = parser.parse_args()

    train_file = Path(args.dataset_location)
    output_dir = Path(args.model_storage_location)

    train_data = get_data(train_file)

    model, tokenizer = fine_tune_model(
        text=train_data,
        epochs=args.epochs,
    )

    save_model(model, tokenizer, output_dir)


if __name__ == "__main__":
    main()
