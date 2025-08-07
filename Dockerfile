FROM python:3.13.5-bullseye

WORKDIR /app

COPY requirements.txt .
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
