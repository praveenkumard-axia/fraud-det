# Pod 1: Data Generator
# High-throughput synthetic transaction generator for FlashBlade

FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    numpy>=1.26.0 \
    pyarrow>=14.0.0 \
    faker>=22.0.0 \
    psutil

COPY gather.py .

ENV OUTPUT_DIR_CPU=/cpu/data/raw
ENV OUTPUT_DIR_GPU=/gpu/data/raw
ENV NUM_WORKERS=16
ENV DURATION_SECONDS=60
ENV MAX_ROWS=1000000
ENV CHUNK_SIZE=100000
ENV FRAUD_RATE=0.005
ENV PYTHONUNBUFFERED=1

CMD ["python", "gather.py"]