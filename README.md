# Brain Tumor MRI Classification

A production-ready, Dockerized FastAPI application that classifies brain tumor MRI images into four categories using a ResNet18 model trained with PyTorch. Built with modular MLOps principles, including model evaluation, API serving, metrics tracking, and containerized deployment.

> 🚧 **Work in Progress** – this project is actively being developed

## 🧪 Dataset

The dataset used is from [Kaggle: Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset?resource=download)

## Features

- ResNet18 classifier trained on brain MRI scans
- Clean modular code: dataset.py, train.py, evaluate.py, api.py
- /metrics endpoint with classification report and confusion matrix
- FastAPI inference API (/predict, /health, /metrics)
- Docker + Gunicorn for production-grade deployment
- Local dev with docker-compose
- Ready for MLflow/W&B tracking and CI/CD pipelines

## Model Info

- Model: ResNet18 fine-tuned on 4 tumor classes
- Dataset: Brain Tumor MRI dataset with Training/ and Testing/ folders
- Input: JPEG brain scan
- Output: Tumor class + confidence score

## API Endpoints

- POST (/predict): Upload MRI image for prediction
- GET (/metrics): Get classification metrics
- GET (/health): Health check (uptime status)

## Example Usage

#### Predict Tumor Type

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@example.jpg"
```

#### Response:

```bash
{
  "predicted_class": "glioma",
  "confidence": 0.9783,
  "class_index": 0
}
```

#### Metrics

```bash
curl http://localhost:8000/metrics
```

## Dockerized Setup

Build and Run with Docker Compose

```bash
docker-compose up --build
```

Visit:

- http://localhost:8000/docs – Swagger UI
- http://localhost:8000/health – Health check
- http://localhost:8000/predict – Upload image

#### Dockerfile

Production-ready with gunicorn and uvicorn:

```bash
CMD ["gunicorn" "src.api:app" "-k" "uvicorn.workers.UvicornWorker" "--bind" "0.0.0.0:8000"]
```

## Project Structure

```bash
.
├── .dvc/               # dvc folder for tracking data
├── data/               # Raw dataset (not tracked)
├── models/             # Trained model weights (.pt)
├── notebooks/          # Notebooks for EDA, training and evaluating the model
├── src/
│   ├── api.py          # FastAPI app
│   ├── dataset.py      # Dataset + transforms
│   ├── evaluate.py     # Evaluation + prediction functions
│   ├── utils.py        # Helpers: accuracy, logging, seeding
│   ├── train.py        # Training script
├── Dockerfile          # Production image
├── docker-compose.yml  # Dev container setup
├── .dockerignore
├── .gitignore
├── .python-version
├── pyproject.toml      # UV .toml file for dependencies
├── uv.lock             # UV file
├── data.dvc
├── requirements.txt
└── README.md           # This file
```
