# Docker Setup for Jetson Orin NX

Complete guide for deploying YOLOE-26 benchmarking and inference on Jetson using Docker.

---

## Why Docker on Jetson?

**Benefits:**
- ✅ **Reproducible environment** - Same setup across development and production
- ✅ **Easy deployment** - Build once, run anywhere
- ✅ **Dependency isolation** - No conflicts with system packages
- ✅ **NVIDIA support** - Official Jetson containers with CUDA/TensorRT pre-installed
- ✅ **Version control** - Docker images are versioned and shareable
- ✅ **Clean rollback** - Easy to revert to previous versions

**NVIDIA provides official containers:**
- `nvcr.io/nvidia/l4t-pytorch` - PyTorch + CUDA + TensorRT
- `nvcr.io/nvidia/l4t-tensorflow` - TensorFlow + CUDA
- `nvcr.io/nvidia/l4t-ml` - Full ML stack (PyTorch, TensorFlow, etc.)

---

## Prerequisites

- Jetson Orin NX with JetPack 5.x or 6.x installed
- Internet connection
- SSH access to Jetson (or direct terminal access)

---

## Installation Steps

### 1. Install Docker on Jetson

```bash
# Update package list
sudo apt-get update

# Install Docker
sudo apt-get install -y docker.io

# Start and enable Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Verify installation
docker --version
```

### 2. Add User to Docker Group

Avoid needing `sudo` for every Docker command:

```bash
# Add current user to docker group
sudo usermod -aG docker $USER

# Apply group changes (or logout/login)
newgrp docker

# Test - should work without sudo
docker ps
```

### 3. Install NVIDIA Container Runtime

Required for GPU access inside containers:

```bash
# Install NVIDIA container toolkit
sudo apt-get install -y nvidia-container-runtime

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker to apply changes
sudo systemctl restart docker
```

### 4. Verify GPU Access in Docker

```bash
# Test GPU access in container
docker run --rm --runtime nvidia nvcr.io/nvidia/l4t-base:r35.2.1 nvidia-smi

# Should show GPU information
# If this works, you're ready to go!
```

---

## Project Dockerfile

Create a `Dockerfile` in your project root:

```dockerfile
# Use NVIDIA's official L4T PyTorch container
# Check your JetPack version and adjust tag accordingly
# JetPack 5.x: r35.x.x
# JetPack 6.x: r36.x.x
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    ultralytics \
    opencv-python-headless \
    numpy \
    pandas \
    tqdm

# Copy project files
COPY segment_road.py /workspace/
COPY benchmark.py /workspace/
COPY requirements.txt /workspace/
COPY scripts/ /workspace/scripts/

# Create necessary directories
RUN mkdir -p /workspace/output /workspace/benchmark

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Default command (interactive shell)
CMD ["/bin/bash"]
```

### Alternative: Minimal Dockerfile

If you want to install dependencies from `requirements.txt`:

```dockerfile
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of project
COPY . /workspace/

# Create output directory
RUN mkdir -p /workspace/output

ENV PYTHONUNBUFFERED=1

CMD ["/bin/bash"]
```

---

## Building the Docker Image

```bash
# Navigate to project directory
cd ~/Documents/jetson-benchmark

# Build the image (takes 5-10 minutes first time)
docker build -t jetson-yolo26:latest .

# Verify image was created
docker images | grep jetson-yolo26
```

**Build with specific tag:**
```bash
docker build -t jetson-yolo26:v1.0 .
```

---

## Running the Container

### Interactive Mode (Development)

```bash
# Run container with GPU access and volume mounts
docker run -it --rm --runtime nvidia \
  -v $(pwd)/benchmark:/workspace/benchmark \
  -v $(pwd)/output:/workspace/output \
  jetson-yolo26:latest \
  /bin/bash

# Now you're inside the container
# Run commands as normal:
python benchmark.py all --model yolo26 --model-size n
```

### One-Shot Command (Production)

```bash
# Run benchmark directly
docker run --rm --runtime nvidia \
  -v $(pwd)/benchmark:/workspace/benchmark \
  -v $(pwd)/output:/workspace/output \
  jetson-yolo26:latest \
  python benchmark.py all --model yolo26 --model-size n --yolo-backend tensorrt
```

### Run Benchmark Script

```bash
# Run the full config benchmark
docker run --rm --runtime nvidia \
  -v $(pwd)/benchmark:/workspace/benchmark \
  -v $(pwd)/output:/workspace/output \
  -v $(pwd)/scripts:/workspace/scripts \
  jetson-yolo26:latest \
  bash scripts/run_all_configs.sh
```

---

## Docker Command Breakdown

```bash
docker run \
  -it \                    # Interactive terminal
  --rm \                   # Remove container after exit
  --runtime nvidia \       # Enable GPU access
  -v $(pwd)/data:/workspace/data \  # Mount local directory
  jetson-yolo26:latest \   # Image name
  /bin/bash                # Command to run
```

**Flags explained:**
- `-it` - Interactive terminal (for shell access)
- `--rm` - Auto-remove container when it exits (keeps system clean)
- `--runtime nvidia` - **Critical** - Enables GPU access
- `-v host:container` - Mount directories (share files between host and container)
- `--name mycontainer` - Give container a name (optional)
- `-p 8080:8080` - Port mapping (if running web services)

---

## Common Workflows

### 1. Development Workflow

```bash
# Start interactive container
docker run -it --rm --runtime nvidia \
  -v $(pwd):/workspace \
  jetson-yolo26:latest

# Inside container:
python benchmark.py all --model yolo26 --model-size n
python segment_road.py  # Test individual components
exit
```

### 2. Benchmark Workflow

```bash
# Run full benchmark suite
docker run --rm --runtime nvidia \
  -v $(pwd)/benchmark:/workspace/benchmark \
  -v $(pwd)/output:/workspace/output \
  jetson-yolo26:latest \
  bash scripts/run_all_configs.sh

# Results will be in output/config-tests/ on host
```

### 3. Single Inference Test

```bash
# Test on single image
docker run --rm --runtime nvidia \
  -v $(pwd):/workspace \
  jetson-yolo26:latest \
  python -c "
from segment_road import YOLO26Segmentor
seg = YOLO26Segmentor(size='n', backend='tensorrt')
result, time_ms = seg.infer('benchmark/images/test.jpg')
print(f'Inference: {time_ms:.2f}ms')
"
```

---

## Managing Docker Images and Containers

### List Images
```bash
docker images
```

### Remove Image
```bash
docker rmi jetson-yolo26:latest
```

### List Running Containers
```bash
docker ps
```

### List All Containers (including stopped)
```bash
docker ps -a
```

### Stop Container
```bash
docker stop <container_id>
```

### Remove Container
```bash
docker rm <container_id>
```

### Clean Up Unused Resources
```bash
# Remove stopped containers, unused images, etc.
docker system prune -a

# Free up space (careful - removes everything not in use)
docker system prune -a --volumes
```

---

## Sharing Your Docker Image

### Save Image to File
```bash
# Export image
docker save jetson-yolo26:latest | gzip > jetson-yolo26.tar.gz

# Transfer to another Jetson
scp jetson-yolo26.tar.gz user@other-jetson:/tmp/

# Load on other Jetson
docker load < jetson-yolo26.tar.gz
```

### Push to Docker Hub
```bash
# Tag image
docker tag jetson-yolo26:latest yourusername/jetson-yolo26:latest

# Login to Docker Hub
docker login

# Push image
docker push yourusername/jetson-yolo26:latest

# Pull on another Jetson
docker pull yourusername/jetson-yolo26:latest
```

---

## Troubleshooting

### GPU Not Accessible

**Error:** `could not select device driver "" with capabilities: [[gpu]]`

**Fix:**
```bash
# Reinstall NVIDIA container runtime
sudo apt-get install -y nvidia-container-runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test again
docker run --rm --runtime nvidia nvcr.io/nvidia/l4t-base:r35.2.1 nvidia-smi
```

### Permission Denied

**Error:** `permission denied while trying to connect to the Docker daemon socket`

**Fix:**
```bash
sudo usermod -aG docker $USER
newgrp docker
# Or logout and login again
```

### Out of Disk Space

**Error:** `no space left on device`

**Fix:**
```bash
# Check disk usage
df -h

# Clean up Docker
docker system prune -a

# Remove old images
docker images
docker rmi <image_id>
```

### Container Can't Find Files

**Issue:** Mounted volumes are empty

**Fix:**
```bash
# Use absolute paths
docker run -v /home/nvidia/Documents/jetson-benchmark:/workspace ...

# Or use $(pwd)
cd ~/Documents/jetson-benchmark
docker run -v $(pwd):/workspace ...
```

### TensorRT Engine Not Found

**Issue:** Engine files not persisting between runs

**Fix:**
```bash
# Mount a persistent volume for model weights
docker run --runtime nvidia \
  -v $(pwd)/models:/root/.cache/ultralytics \
  -v $(pwd):/workspace \
  jetson-yolo26:latest
```

---

## Best Practices

1. **Use .dockerignore**
   ```
   # .dockerignore
   output/
   .git/
   __pycache__/
   *.pyc
   .venv/
   *.engine
   ```

2. **Layer Caching**
   - Copy `requirements.txt` before other files
   - Install dependencies before copying code
   - Speeds up rebuilds

3. **Multi-Stage Builds** (Advanced)
   ```dockerfile
   # Build stage
   FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3 AS builder
   COPY requirements.txt .
   RUN pip install --user -r requirements.txt
   
   # Runtime stage
   FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
   COPY --from=builder /root/.local /root/.local
   COPY . /workspace
   ```

4. **Version Everything**
   - Tag images with version numbers
   - Pin dependency versions in requirements.txt
   - Document which JetPack version you're using

5. **Keep Images Small**
   - Use `--no-cache-dir` with pip
   - Clean up apt cache: `rm -rf /var/lib/apt/lists/*`
   - Remove unnecessary files

---

## Docker Compose (Optional)

For more complex setups, use `docker-compose.yml`:

```yaml
version: '3.8'

services:
  benchmark:
    build: .
    runtime: nvidia
    volumes:
      - ./benchmark:/workspace/benchmark
      - ./output:/workspace/output
    environment:
      - CUDA_VISIBLE_DEVICES=0
    command: bash scripts/run_all_configs.sh
```

Run with:
```bash
docker-compose up
```

---

## Summary

**Quick Start:**
```bash
# 1. Install Docker + NVIDIA runtime (one-time setup)
sudo apt-get install -y docker.io nvidia-container-runtime
sudo usermod -aG docker $USER
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 2. Build your image
cd ~/Documents/jetson-benchmark
docker build -t jetson-yolo26 .

# 3. Run benchmarks
docker run --rm --runtime nvidia \
  -v $(pwd)/benchmark:/workspace/benchmark \
  -v $(pwd)/output:/workspace/output \
  jetson-yolo26 \
  bash scripts/run_all_configs.sh
```

**Your mentor was right - Docker is the way to go for Jetson deployment!**
