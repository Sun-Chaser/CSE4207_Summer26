# Python Basic Libraries
import argparse
import re
from pathlib import Path

# PyTorch and Transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# change me
def load_model(model_location):
    model_path = Path(model_location)

    if not model_path.exists() or not model_path.is_dir():
        raise ValueError("Model not found.")

    return AutoTokenizer.from_pretrained(model_path), AutoModelForCausalLM.from_pretrained(model_path)

def clean_answer(answer):
    return re.sub(r"\s+", " ", answer).strip()

def answer_question(model, tokenizer, question):

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

def answer_questions(model, tokenizer):
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
        answer_question(model, tokenizer, args.question)
    else:
        answer_questions(model, tokenizer)

    


if __name__ == "__main__":
    main()
