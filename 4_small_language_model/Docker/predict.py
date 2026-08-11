# Python Basic Libraries
import argparse
import os
import re
from pathlib import Path

# SQL Database
import psycopg2
from psycopg2.extras import Json

# PyTorch and Transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Read from PV
def load_model(model_location):
    model_path = Path(model_location)

    if not model_path.exists() or not model_path.is_dir():
        log_training_event(
            job_type="prediction",
            status="failed",
            payload={"model_name": model_path.name, "error": "The specified model path does not exist or is not a directory."},
        )
        raise ValueError("Model not found.")

    return AutoTokenizer.from_pretrained(model_path), AutoModelForCausalLM.from_pretrained(model_path)

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "slm-service"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "slmDB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        connect_timeout=5,
    )

def log_training_event(job_type, status, payload):
    try:
        with get_db_connection() as connection:
            with connection.cursor() as cursor:
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
                cursor.execute(
                    """
                    INSERT INTO training_logs (job_type, status, payload)
                    VALUES (%s, %s, %s)
                    """,
                    (job_type, status, Json(payload)),
                )
    except psycopg2.Error as error:
        print(f"Warning: Failed to log metrics to Postgres: {error}")

def clean_answer(answer):
    return re.sub(r"\s+", " ", answer).strip()

def answer_question(model, model_path, tokenizer, question):

    # Set the model to evaluation mode
    model.eval()

    # Tokenize the input question
    inputs = tokenizer(question, return_tensors="pt")

    # Generate an answer using the model
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=256)

    # Decode the generated tokens to get the answer
    answer = clean_answer(tokenizer.decode(outputs[0], skip_special_tokens=True))
    
    print(f"Answer: {answer}")
    log_training_event(
        job_type="one-time-prediction",
        status="completed",
        payload={"model_name": model_path.name, "question": question, "answer": answer},
    )

def answer_questions(model, tokenizer, model_path):
    num_questions_answered = 0  # Initialize the counter for answered questions
    while True:
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break

        # Tokenize the input question
        inputs = tokenizer(question, return_tensors="pt")

        # Generate an answer using the model
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=256)

        # Decode the generated tokens to get the answer
        answer = clean_answer(tokenizer.decode(outputs[0], skip_special_tokens=True))
        print(f"Answer: {answer}\n")

        log_training_event(
            job_type="predict",
            status="running",
            payload={"model_name": model_path.name, "question": question, "answer": answer},
        )
        num_questions_answered += 1  # Increment the counter for answered questions
    
    log_training_event(
        job_type="predict",
        status="completed",
        payload={"model_name": model_path.name, "number_of_questions_answered": num_questions_answered, "message": "Prediction session completed."},
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fetch a fine-tuned Small Language Model and generate answers based on user input question."
        )
    )

    parser.add_argument(
        "--model_location",
        required=True,
        help="PV path to the pre-trained and fine-tuned file folder.",
    )

    args = parser.parse_args()

    model_path = Path(args.model_location)

    if not model_path.exists() or not model_path.is_dir():
        parser.error(f"The specified model path '{model_path}' does not exist or is not a directory.")
        log_training_event(
            job_type="predict",
            status="failed",
            payload={"error": "The model path does not exisit.", "message": "Prediction session exited."},
        )

    print(f"Using model from: {model_path}")
    answer_question(model_path)

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fetch a fine-tuned Small Language Model and generate answers based on user input question."
        )
    )

    parser.add_argument(
        "--model_location",
        required=True,
        help="Path to the pre-trained and fine-tuned file folder.",
    )

    parser.add_argument(
        "--question",
        help="Question to ask the fine-tuned model.",
    )

    args = parser.parse_args()

    tokenizer, model = load_model(args.model_location)

    if args.question:
        answer_question(model, args.model_location, tokenizer, args.question)
    else:
        answer_questions(model, args.model_location, tokenizer)

if __name__ == "__main__":
    main()
