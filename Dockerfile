# Jetson Orin NX Docker image for road segmentation + benchmarking.
# Default base is JetPack 6 / L4T R36.3. Override at build time if needed:
#   docker build --build-arg BASE_IMAGE=<your_l4t_image> -t jetson-benchmark:latest .

ARG BASE_IMAGE=nvcr.io/nvidia/l4t-jetpack:r36.3.0
FROM ${BASE_IMAGE}

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

# Torch 2.6 wheel in wheels/ expects cuDNN 9 symbols.
# These paths are populated by nvidia-cudnn-cu12 and nvidia-cublas-cu12.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH}

# System packages for OpenCV/headless runtime and general tooling.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libopenblas0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy wheel files first so torch install is cached in layers.
COPY wheels/ /workspace/wheels/

# Install Jetson-specific torch + torchvision from local wheels.
# Then install project Python dependencies without pulling a conflicting torch.
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install /workspace/wheels/torch-2.6.0-cp310-cp310-linux_aarch64.whl && \
    python3 -m pip install /workspace/wheels/torchvision-0.21.0-cp310-cp310-linux_aarch64.whl && \
    python3 -m pip install nvidia-cudnn-cu12

# Copy requirements separately for layer caching.
COPY requirements.txt /workspace/requirements.txt
RUN python3 -m pip install -r /workspace/requirements.txt

# Fail build early if key runtime imports are missing.
RUN python3 - <<'PY'
import cv2  # noqa: F401
import ftfy  # noqa: F401
import torch  # noqa: F401
import ultralytics  # noqa: F401
import wcwidth  # noqa: F401
print("python_runtime_imports_ok")
PY

# Copy project files.
COPY . /workspace/

# Ensure expected writable output locations exist.
RUN mkdir -p /workspace/output /workspace/benchmark

CMD ["/bin/bash"]