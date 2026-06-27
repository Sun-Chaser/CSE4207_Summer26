FROM python:3

EXPOSE 5432

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["echo", "Hello from Docker!"]
# CMD ["sh", "ml_pipe.sh"]
