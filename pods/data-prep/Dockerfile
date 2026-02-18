# Pod 2: Feature Engineering
# Multi-GPU RAPIDS for data preparation

FROM nvcr.io/nvidia/rapidsai/base:24.02-cuda12.0-py3.10

ENV RAPIDS_NO_INITIALIZE=1
ENV CUDF_LOGGING_LEVEL=CRITICAL
ENV DASK_DISTRIBUTED__LOGGING__DISTRIBUTED=critical
ENV PYTHONUNBUFFERED=0


RUN pip install --no-cache-dir \
    numpy>=1.26.0 \
    pyarrow>=14.0.0 \
    faker>=22.0.0 \
    psutil \
    prometheus-client \
    pandas \
    polars

WORKDIR /app
COPY prepare.py .

ENV INPUT_DIR=/cpu/data/raw
ENV OUTPUT_DIR=/cpu/data/features
ENV FORCE_CPU=true
ENV BATCH_MODE=true

CMD ["python", "prepare.py"]