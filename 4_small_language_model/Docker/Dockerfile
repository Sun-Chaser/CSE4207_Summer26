FROM python:3.12.8

EXPOSE 5432

ENV HF_HOME=/opt/huggingface \
    HF_HUB_CACHE=/opt/huggingface/hub \
    TRANSFORMERS_CACHE=/opt/huggingface/transformers

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN python3 - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "distilgpt2"
AutoTokenizer.from_pretrained(model_name)
AutoModelForCausalLM.from_pretrained(model_name)
print(f"Cached Hugging Face model: {model_name}")
PY

COPY . .

CMD ["echo", "Hello from Docker!"]
# CMD ["sh", "ml_pipe.sh"]
