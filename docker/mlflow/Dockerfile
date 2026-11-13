# MLflow tracking server for the model pipeline (same MLflow version as pyproject.toml).
FROM python:3.11-slim
RUN pip install --no-cache-dir mlflow==3.18.0
EXPOSE 5000
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts", \
     "--allowed-hosts", "localhost,localhost:5000,localhost:5001,127.0.0.1,127.0.0.1:5000,127.0.0.1:5001,mlflow,mlflow:5000"]
