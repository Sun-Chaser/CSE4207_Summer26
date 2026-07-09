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

def answer_question(model_path):
    model_name = model_path.name  # Use the folder name as the model name

    # Load the fine-tuned model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    print(f"Model '{model_name}' loaded successfully. You can now ask questions.")

    # Set the model to evaluation mode
    model.eval()

    num_questions_answered = 0  # Initialize the counter for answered questions

    while True:
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break

        # Tokenize the input question
        inputs = tokenizer(question, return_tensors="pt")

        # Generate an answer using the model
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=100)

        # Decode the generated tokens to get the answer
        answer = clean_answer(tokenizer.decode(outputs[0], skip_special_tokens=True))
        print(f"Answer: {answer}\n")

        log_training_event(
            job_type="predict",
            status="running",
            payload={"model_name": model_name, "question": question, "answer": answer},
        )
        num_questions_answered += 1  # Increment the counter for answered questions
    
    log_training_event(
        job_type="predict",
        status="completed",
        payload={"model_name": model_name, "number_of_questions_answered": num_questions_answered, "message": "Prediction session completed."},
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


if __name__ == "__main__":
    main()
