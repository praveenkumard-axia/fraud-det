# Pod 3: Model Training
# XGBoost with GPU acceleration

FROM nvcr.io/nvidia/rapidsai/base:24.02-cuda12.0-py3.10

WORKDIR /app

COPY requirements.txt .
# Pin pandas to avoid is_extension_type error in RAPIDS 24.02
# Install torch with CUDA 11.8 support (compat with 12.0 container)
RUN pip install "pandas<2.0.0" && \
    pip install torch --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir -r requirements.txt

COPY train.py .

ENV INPUT_DIR=/gpu/data/features
ENV OUTPUT_DIR_CPU=/cpu/models
ENV OUTPUT_DIR_GPU=/gpu/models
ENV PYTHONUNBUFFERED=1

CMD ["python", "train.py"]