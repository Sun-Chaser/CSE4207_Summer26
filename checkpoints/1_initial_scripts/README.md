# Initial Small Language Model Scripts

This checkpoint demonstrates a minimal, local machine-learning pipeline for fine-tuning a small language model and using the saved model for inference. It uses Hugging Face Transformers and PyTorch to fine-tune `distilgpt2` on a text file containing question-and-answer pairs.

The scripts run on the CPU and do not require Docker, Kubernetes, or cloud storage. They are intended as a simple starting point for the later infrastructure building.

## Files

| File | Purpose |
| --- | --- |
| `fine-tune.py` | Reads question-and-answer examples, fine-tunes `distilgpt2`, and saves the model and tokenizer. |
| `predict.py` | Loads a saved model and generates answers in single-question or interactive mode. |
| `qa_training.txt` | Example training data in the expected format. |
| `requirements.txt` | Python dependencies for the scripts. |

## Prerequisites

- Python 3.12 is recommended.
- An internet connection is required the first time the base model and tokenizer are downloaded from Hugging Face.
- Allow enough free disk space for the Python dependencies, downloaded base
  model, and saved fine-tuned model.

## Setup

From this directory, create and activate a virtual environment:

```bash
python3.12 -m venv .venv_ml
source .venvml/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Training Data Format

The input file must be UTF-8 text containing one or more entries labeled
`Question:` and `Answer:`:

```text
Question: What is cloud computing?
Answer: Cloud computing provides computing resources over a network.

Question: What is a Kubernetes Job?
Answer: A Kubernetes Job runs a task until it completes successfully.
```

The parser supports multi-line questions and answers. Each training example is limited to 128 tokens (or characters). If truncation removes every answer token, the script stops with an error.

## Fine-Tune the Model

Run the following command from this directory:

```bash
python fine-tune.py \
  --dataset_location qa_training.txt \
  --model_storage_location saved_model \
  --epochs 3
```

Arguments:

| Argument | Required | Default | Description |
| --- | --- | --- | --- |
| `--dataset_location` | Yes | — | Path to the question-and-answer training file. |
| `--model_storage_location` | Yes | — | Directory in which to save the trained model and tokenizer. Missing directories are created automatically. |
| `--epochs` | No | `1` | Number of complete passes through the training dataset. |

Training uses a batch size of 1, a learning rate of `5e-5`, and the CPU. The
script prints the number of examples and average loss for each epoch. When it
finishes, `saved_model/` contains the Hugging Face model and tokenizer files
needed for inference.

## Generate an Answer

To ask one question and exit:

```bash
python predict.py \
  --model_location saved_model \
  --question 'Question: What is Docker?'
```

The `Question: ...` structure matches the prompts used during training. The script accepts any input string, but using the training format usually gives the model the clearest continuation prompt. 

The choice of `distilgpt2` may not give any answer at all due to its small size. However, the test on more powerful models like `llama3-2B` will return a reasonable answer but Kubernetes resource won't allow reasonable time to tune this. Students are free to explore different models for final project. Notice that a Hugging Face model like `llama3` would require application for access and personal token as needed.

Arguments:

| Argument | Required | Default | Description |
| --- | --- | --- | --- |
| `--model_location` | Yes | — | Directory containing the saved Hugging Face model and tokenizer. |
| `--question` | No | — | One prompt to process before the script exits. |

If `--question` is omitted, the script starts an interactive loop:

```bash
python predict.py --model_location saved_model
```

Enter prompts one at a time and type `exit` to stop. In interactive mode, an
example prompt is:

```text
Question: What is Kubernetes?
Answer:
```

Generation is limited to 256 total tokens. Because `distilgpt2` is a small
general-purpose model and the sample dataset is tiny, responses may be
inaccurate, repetitive, or include the original prompt. This checkpoint is
meant to illustrate the pipeline rather than produce a production-quality
question-answering system.

## Troubleshooting

- **`Dataset file not found.`** Check the path passed to
  `--dataset_location`.
- **No question-and-answer pairs were found.** Ensure the data uses the exact
  `Question:` and `Answer:` labels.
- **`Model not found.`** Check that `--model_location` points to the directory
  created by `fine-tune.py`.
- **The initial run is slow.** The first training run downloads `distilgpt2`,
  and all training in this checkpoint runs on the CPU.
- **The generated answer is poor.** Add more representative training examples,
  train for additional epochs, and use the same prompt structure at inference
  time. More epochs can also cause overfitting on a small dataset.
