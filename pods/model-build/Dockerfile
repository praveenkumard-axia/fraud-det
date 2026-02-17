# Pod 3: Model Training
# XGBoost with GPU acceleration

FROM nvcr.io/nvidia/rapidsai/base:24.02-cuda12.0-py3.10

WORKDIR /app

RUN pip install --no-cache-dir xgboost>=2.0.0 \
    numpy>=1.26.0 \
    pyarrow>=14.0.0 \
    faker>=22.0.0 \
    psutil \
    prometheus-client \
    polars \
    pandas 

COPY train.py .

ENV INPUT_DIR=/gpu/data/features
ENV OUTPUT_DIR_CPU=/cpu/models
ENV OUTPUT_DIR_GPU=/gpu/models
ENV PYTHONUNBUFFERED=1

CMD ["python", "train.py"]