# 3.12 is good for the transformer and torch
FROM python:3.12-slim

# Set as needed
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY job_templates/ ./job_templates/

EXPOSE 8080

# Set up submitter endpoint
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
