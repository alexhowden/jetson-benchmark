# Docker Setup for Jetson Orin NX

This guide sets up Docker for this project on Jetson and builds an image that installs PyTorch and TorchVision from local wheel files in `wheels/`.

Verified on Jetson Orin NX with JetPack 6 / L4T R36.3.

## Project Layout Assumptions

- Model files are in `models/`
- PyTorch wheels are in `wheels/`
  - `torch-2.6.0-cp310-cp310-linux_aarch64.whl`
  - `torchvision-0.21.0-cp310-cp310-linux_aarch64.whl`
- Dockerfile is in project root

## 1. Install and Enable Docker

If Docker is not already installed:

```bash
sudo apt-get update
sudo apt-get install -y docker.io nvidia-container-runtime
sudo systemctl enable --now docker
```

Add your user to Docker group (recommended):

```bash
sudo usermod -aG docker "$USER"
newgrp docker
```

If your shell still cannot access Docker socket, use non-interactive sudo for commands:

```bash
printf 'nvidia\n' | sudo -S docker info
```

## 2. Configure NVIDIA Runtime

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## 3. Verify GPU Access in a Container

On many Jetsons, `nvidia-smi` is unavailable. Use `deviceQuery` style checks via CUDA libs if needed.

Quick runtime check:

```bash
printf 'nvidia\n' | sudo -S docker run --rm --runtime nvidia nvcr.io/nvidia/l4t-base:r36.3.0 echo "NVIDIA runtime OK"
```

## 4. Build the Project Image

From project root:

```bash
printf 'nvidia\n' | sudo -S docker build -t jetson-benchmark:latest .
```

If build fails with an iptables/bridge error like `can't initialize iptables table raw`, use host networking:

```bash
printf 'nvidia\n' | sudo -S docker build --network host -t jetson-benchmark:latest .
```

Freeze-safe option (recommended in VS Code for long Jetson builds):

```bash
LOG=/tmp/jetson-benchmark-docker-build.log
printf 'nvidia\n' | sudo -S nohup docker build --network host -t jetson-benchmark:latest . > "$LOG" 2>&1 < /dev/null &
echo "Build started. Log: $LOG"
```

Check progress without streaming everything into VS Code:

```bash
tail -n 60 /tmp/jetson-benchmark-docker-build.log
```

Confirm image built:

```bash
printf 'nvidia\n' | sudo -S docker images | grep jetson-benchmark
```

What this image does:

- Starts from NVIDIA L4T JetPack base (`nvcr.io/nvidia/l4t-jetpack:r36.3.0`)
- Installs system deps for OpenCV/headless usage
- Installs torch + torchvision from `wheels/`
- Installs project deps from `requirements.txt` with dependency resolution enabled
- Copies repo content into `/workspace`

To use a different base image (future JetPack/L4T):

```bash
printf 'nvidia\n' | sudo -S docker build \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/l4t-jetpack:r36.4.0 \
  -t jetson-benchmark:latest .
```

## 5. Run the Container

Interactive shell:

```bash
printf 'nvidia\n' | sudo -S docker run -it --rm --runtime nvidia \
  -v "$(pwd)":/workspace \
  -w /workspace \
  jetson-benchmark:latest
```

If run fails with an iptables/bridge error like `can't initialize iptables table raw`, add `--network host`:

```bash
printf 'nvidia\n' | sudo -S docker run -it --rm --runtime nvidia --network host \
  -v "$(pwd)":/workspace \
  -w /workspace \
  jetson-benchmark:latest
```

Run benchmark command directly:

```bash
printf 'nvidia\n' | sudo -S docker run --rm --runtime nvidia \
  -v "$(pwd)":/workspace \
  -w /workspace \
  jetson-benchmark:latest \
  python3 benchmark.py all --model yolo26 --model-size n
```

With host-network fallback:

```bash
printf 'nvidia\n' | sudo -S docker run --rm --runtime nvidia --network host \
  -v "$(pwd)":/workspace \
  -w /workspace \
  jetson-benchmark:latest \
  python3 benchmark.py all --model yolo26 --model-size n
```

Run segmentation command directly:

```bash
printf 'nvidia\n' | sudo -S docker run --rm --runtime nvidia \
  -v "$(pwd)":/workspace \
  -w /workspace \
  jetson-benchmark:latest \
  python3 segment_road.py --model yolo26 --input benchmark/images/raw
```

With host-network fallback:

```bash
printf 'nvidia\n' | sudo -S docker run --rm --runtime nvidia --network host \
  -v "$(pwd)":/workspace \
  -w /workspace \
  jetson-benchmark:latest \
  python3 segment_road.py --model yolo26 --input benchmark/images/raw
```

## 6. Notes for Future Jetson Devices

- Keep wheel filenames in `wheels/` synced with the `Dockerfile` install commands.
- If wheel filenames change, update both:
  - `Dockerfile`
  - this document
- Use an L4T base image that matches JetPack on the target device.
- If you must run Docker commands with sudo in automation/non-interactive shells, use:

```bash
printf 'YOUR_PASSWORD\n' | sudo -S <command>
```

Example:

```bash
printf 'nvidia\n' | sudo -S systemctl restart docker
```

## 7. Troubleshooting

### Permission denied on Docker socket

```bash
sudo usermod -aG docker "$USER"
newgrp docker
```

### NVIDIA runtime not found

```bash
sudo apt-get install -y nvidia-container-runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Build fails on wheel install

Check filenames:

```bash
ls -lh wheels/
```

Then ensure they match `Dockerfile` exactly.

### Runtime dependency mismatch after updates

If Dockerfile or requirements changed, rebuild the image before running benchmarks:

```bash
printf 'nvidia\n' | sudo -S docker build --network host -t jetson-benchmark:latest .
```

Optional quick runtime check:

```bash
printf 'nvidia\n' | sudo -S docker run --rm --runtime nvidia --network host \
  jetson-benchmark:latest \
  python3 -c "import torch, cv2, ultralytics; print('container_runtime_ok')"
```
