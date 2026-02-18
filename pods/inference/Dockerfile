FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    numpy>=1.26.0 \
    polars>=0.20.0 \
    xgboost>=2.0.0 \
    psutil>=5.9.0 \
    tritonclient \
    prometheus-client

COPY client.py .

ENV MODEL_DIR=/cpu/models/fraud_xgboost
ENV DATA_DIR=/cpu/data/features
ENV EXECUTION_TYPE=CPU
ENV PYTHONUNBUFFERED=0

CMD ["python", "client.py"]
