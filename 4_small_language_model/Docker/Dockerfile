FROM python:3.12.8

EXPOSE 5432

# Set up Hugging Face cache directories (otherwise, the model will be downloaded every time the container is built)
ENV HF_HOME=/opt/huggingface \
    HF_HUB_CACHE=/opt/huggingface/hub \
    TRANSFORMERS_CACHE=/opt/huggingface/transformers

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache the Hugging Face model and tokenizer to avoid downloading them at runtime
RUN python3 - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "distilgpt2"
AutoTokenizer.from_pretrained(model_name)
AutoModelForCausalLM.from_pretrained(model_name)
print(f"Cached Hugging Face model: {model_name}")
PY

COPY . .

# No need for CMD (will be overwritten by K8S)
CMD ["echo", "Hello from Docker!"]