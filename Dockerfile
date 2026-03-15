FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MODELS_DIR=/app/models \
    HF_HUB_CACHE=/app/models/hf_cache

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-dev git ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Clone Seed-VC source (modules, configs, hf_utils)
RUN git clone --depth 1 https://github.com/Plachtaa/seed-vc.git /app/seed-vc

# Pre-download models into image
COPY download_models.py .
RUN python3 download_models.py

# Copy worker code
COPY voice_converter_serverless.py .
COPY audio_processor.py .
COPY handler.py .

# Copy voice presets
COPY presets/ /app/presets/

# Add seed-vc to Python path
ENV PYTHONPATH="/app/seed-vc:${PYTHONPATH}"

CMD ["python3", "handler.py"]
