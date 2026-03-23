# Off-Road Traversable Road Segmentation

## Quick Comparison

|                | YOLO26x (YOLOE-26x)                    | SAM3                                   | SAM2.1-L                               |
| -------------- | -------------------------------------- | -------------------------------------- | -------------------------------------- |
| **Approach**   | Real-time open-vocabulary segmentation | Foundation model, concept segmentation | Point-prompted foundation segmentation |
| **Speed**      | ~3–8 ms / image (H200)                 | ~30 ms / image (H200)                  | ~50–100 ms / image (H200)              |
| **Model size** | ~70 MB                                 | 3.4 GB                                 | ~224 MB                                |
| **Prompting**  | Text prompts (open-vocab)              | Text prompts (open-vocab)              | Point prompt (bottom-centre of frame)  |
| **Zero-shot**  | Good — open-vocab via text prompts     | Strong — 47.0 LVIS Mask AP             | Strong — SAM2.1 architecture           |
| **Video**      | Frame-by-frame                         | Temporal tracking built-in             | Frame-by-frame                         |
| **Weights**    | Auto-downloaded                        | Manual download required               | Auto-downloaded                        |

---

## 1 — Environment Setup (Jetson venv)

### 1a. Create and activate a venv

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 1b. Install Jetson PyTorch (CUDA 12.2)

Use NVIDIA's Jetson PyTorch thread to get the exact wheel URL and wheel filename for your JetPack/CUDA/Python combo:

- https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

For JetPack 6.x (CUDA 12.2), torch 2.3.0, use this flow:

```bash
# 1) install prerequisites
sudo apt-get update
sudo apt-get install -y libopenblas-base libopenmpi-dev libomp-dev libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
pip3 install --upgrade pip
pip3 install 'Cython<3' numpy

# 2) install PyTorch 2.3.0 wheel
wget https://nvidia.box.com/shared/static/47f7220m09sh0af89p67u7vsu8id98be.whl -O torch-2.3.0-cp310-cp310-linux_aarch64.whl
pip3 install torch-2.3.0-cp310-cp310-linux_aarch64.whl

# 3) install torchvision 0.18.0 from source
git clone --branch v0.18.0 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.18.0
python3 setup.py install --user
cd ..

# 4) install torchaudio 2.3.0 wheel
wget https://nvidia.box.com/shared/static/5412702z8vubun372138x1bms9xuncl9.whl -O torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl
pip3 install torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl

# 5) verify versions and CUDA availability
python3 -c "import torch, torchvision, torchaudio; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('torchvision version:', torchvision.__version__); print('torchaudio version:', torchaudio.__version__)"
```

Notes:

- These wheel links come from the NVIDIA Jetson PyTorch forum post for torch 2.3.0 on JetPack 6.x.
- If NVIDIA rotates links or updates wheel names, use the latest links from the forum thread above.
- If you use the project venv from section 1a, use venv executables instead of system ones:

```bash
python -m pip install --upgrade pip
python -m pip install 'Cython<3' numpy
python -m pip install torch-2.3.0-cp310-cp310-linux_aarch64.whl
python -m pip install torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl
python -c "import torch, torchvision, torchaudio; print(torch.__version__, torchvision.__version__, torchaudio.__version__, torch.cuda.is_available())"
```

### 1c. Install remaining dependencies without replacing torch

Some pip installs may try to replace Jetson-specific torch builds. If that happens, install dependencies with no dependency resolution and then add missing packages manually as needed:

```bash
pip install --no-deps -r requirements.txt
```

If torch gets replaced accidentally, reinstall torch from the NVIDIA Jetson thread above.

> **Note on the CLIP package (SAM3 only):**
> If you already have a different `clip` package installed, replace it:
>
> ```bash
> pip uninstall clip -y
> pip install git+https://github.com/ultralytics/CLIP.git
> ```

---

## 2 — Download Model Weights

### YOLO26 (YOLOE-26)

Weights are **downloaded automatically** on first run. Nothing to do.

### SAM2.1-L

Weights are **downloaded automatically** by Ultralytics on first run (~224 MB). Nothing to do.

### SAM3

SAM3 weights require manual download:

1. Go to <https://huggingface.co/facebook/sam3> and click **"Request access"**.
2. Once approved, go to the `Files and versions` tab and download **`sam3.pt`** (~3.4 GB).
3. Place `sam3.pt` in the project root (same folder as `segment_road.py`).

---

## 3 — Running the Script

### Segment all test images and videos

```bash
python segment_road.py --model yolo26 --input test_data/
python segment_road.py --model sam21  --input test_data/
python segment_road.py --model sam3   --input test_data/
```

### Single image

```bash
python segment_road.py --model yolo26 --input test_data/orfd.png
python segment_road.py --model sam21  --input test_data/orfd.png
python segment_road.py --model sam3   --input test_data/orfd.png
```

### Single video

```bash
python segment_road.py --model yolo26 --input test_data/aa.mp4
python segment_road.py --model sam21  --input test_data/aa.mp4
python segment_road.py --model sam3   --input test_data/aa.mp4
```

### Video writer backend (end-to-end output path)

```bash
# auto: uses GStreamer/NVENC on Jetson when available, falls back to OpenCV
python segment_road.py --model yolo26 --input test_data/aa.mp4 --video-writer auto

# force OpenCV writer
python segment_road.py --model yolo26 --input test_data/aa.mp4 --video-writer opencv

# force GStreamer writer
python segment_road.py --model yolo26 --input test_data/aa.mp4 --video-writer gstreamer
```

### Video capture backend (GPU decode on Jetson)

```bash
# auto: uses GStreamer/NVDEC on Jetson when available, falls back to OpenCV
python segment_road.py --model yolo26 --input test_data/aa.mp4 --video-capture auto

# force OpenCV capture
python segment_road.py --model yolo26 --input test_data/aa.mp4 --video-capture opencv

# force GStreamer capture
python segment_road.py --model yolo26 --input test_data/aa.mp4 --video-capture gstreamer
```

### YOLO TensorRT backend

```bash
# Use TensorRT backend (exports engine on first run if needed)
python segment_road.py --model yolo26 --input test_data/ --yolo-backend tensorrt

# Use a prebuilt TensorRT engine explicitly
python segment_road.py --model yolo26 --input test_data/ \
  --yolo-backend tensorrt --yolo-engine /path/to/yoloe-26x-seg.engine
```

### Save a JSON metrics report

```bash
python segment_road.py --model yolo26 --input test_data/ --report
python segment_road.py --model sam21  --input test_data/ --report
python segment_road.py --model sam3   --input test_data/ --report
```

Reports are saved as `output/<model>/metrics_<model>.json`.

### CLI Options

| Flag              | Default       | Description                                                           |
| ----------------- | ------------- | --------------------------------------------------------------------- |
| `--model`         | _(required)_  | `yolo26`, `sam21`, or `sam3`                                          |
| `--input`         | `test_data/`  | Image/video file or directory                                         |
| `--output`        | `output/`     | Base output directory — a `<model>` subfolder is created inside       |
| `--conf`          | `0.25`        | Detection confidence threshold                                        |
| `--prompts`       | built-in set  | Custom text prompts _(YOLO26 and SAM3 only — ignored for SAM2.1)_     |
| `--model-size`    | `x`           | YOLOE-26 size: `n` / `s` / `m` / `l` / `x` _(ignored for SAM models)_ |
| `--yolo-weights`  | auto          | Optional YOLO weights/engine path                                     |
| `--yolo-backend`  | `pytorch`     | YOLO backend: `pytorch` / `tensorrt`                                  |
| `--yolo-engine`   | none          | TensorRT engine path (used with `--yolo-backend tensorrt`)            |
| `--sam3-weights`  | `sam3.pt`     | Path to SAM3 weights _(ignored for YOLO26 and SAM2.1)_                |
| `--sam21-weights` | `sam2.1_l.pt` | Path to SAM2.1 weights — auto-downloaded if not present               |
| `--video-writer`  | `auto`        | Video output backend: `auto` / `opencv` / `gstreamer`                 |
| `--video-capture` | `auto`        | Video input backend: `auto` / `opencv` / `gstreamer`                  |
| `--report`        | off           | Save JSON metrics report to the model output directory                |

### Model and Flag Compatibility

| Option            | yolo26 | sam3 | sam21 |
| ----------------- | ------ | ---- | ----- |
| `--prompts`       | Yes    | Yes  | No    |
| `--model-size`    | Yes    | No   | No    |
| `--yolo-weights`  | Yes    | No   | No    |
| `--yolo-backend`  | Yes    | No   | No    |
| `--yolo-engine`   | Yes    | No   | No    |
| `--sam3-weights`  | No     | Yes  | No    |
| `--sam21-weights` | No     | No   | Yes   |
| `--video-capture` | Yes    | Yes  | Yes   |
| `--video-writer`  | Yes    | Yes  | Yes   |

Notes:

- `--yolo-backend tensorrt` and `--yolo-engine` are YOLO-only in this repo.
- SAM models currently run through their existing Ultralytics/PyTorch path.

### Custom text prompts (YOLO26 / SAM3 only)

```bash
python segment_road.py --model yolo26 --input test_data/ \
    --prompts "dirt trail" "mud road" "driveable terrain" "gravel path"
```

---

## 4 — Output

All outputs are written to a model-specific subfolder inside `--output`:

```
output/
  yolo26/
  sam21/
  sam3/
```

### Annotated images

`output/<model>/<filename>_<model>_road.<ext>`
— Green semi-transparent mask over the traversable road
— Cyan contour boundary
— HUD overlay with per-frame metrics

### Annotated videos

`output/<model>/<filename>_<model>_road.mp4`
— Same overlay per frame + per-frame HUD

---

## 5 — Metrics

This repository tracks **benchmark metrics + thresholds** so you can compare models apples-to-apples on:

- **200 Images** (with labels)
- **17 Videos** (no ground truth required)
- **Jetson performance** (NRU-51V+ / Jetson Orin NX 16GB targets)

Models under test:

- **SAM3**: `sam3.pt` (local weights)
- **SAM2.1-L**: `sam2.1_l.pt` (local weights)
- **FastSAM**: `FastSAM-x.pt` (local weights)
- **MobileSAM**: `mobile_sam.pt` (local weights)
- **YOLOE-26 Seg**: `yoloe-26{n|s|m|l|x}-seg.pt` (local weights)

### Performance (always reported)

| Metric              | How it is calculated                                  |
| ------------------- | ----------------------------------------------------- |
| `inference_time_ms` | `time.perf_counter()` around `model.predict()`, in ms |
| `fps`               | `1000 / inference_time_ms`                            |

#### Jetson Performance targets (NRU-51V+ / Jetson Orin NX 16GB)

| Metric                | Threshold |
| --------------------- | --------- |
| GPU Utilization       | ≤ 95%     |
| CPU Utilization       | ≤ 80%     |
| RAM Usage             | ≤ 80%     |
| Image Inference Speed | ≤ 200 ms  |
| Video Inference Speed | ≤ 25 ms   |
| Video FPS             | ≥ 25 FPS  |

#### Video benchmark thresholds (10 videos)

| Metric                   | Threshold |
| ------------------------ | --------- |
| Temporal IoU             | ≥ 0.85    |
| No-detection Rate        | ≤ 0.01    |
| Frame-to-frame Stability | ≥ 0.90    |

#### Image benchmark thresholds (200 images)

| Metric              | Threshold |
| ------------------- | --------- |
| mIoU / IoU          | ≥ 0.85    |
| F1 Score            | ≥ 0.90    |
| Pixel Accuracy      | ≥ 0.90    |
| False Negative Rate | ≤ 0.08    |
| False Positive Rate | ≤ 0.08    |

---

## 6 — Benchmark Automation

Run all configured models in one command:

```bash
bash scripts/run_all_benchmarks.sh
```

Useful options:

```bash
# Show available flags
bash scripts/run_all_benchmarks.sh --help

# Use OpenCV writer instead of auto
bash scripts/run_all_benchmarks.sh --video-writer opencv

# Use GPU decode/capture path
bash scripts/run_all_benchmarks.sh --video-capture gstreamer

# Run only YOLO TensorRT path (no PyTorch YOLO baseline)
bash scripts/run_all_benchmarks.sh --yolo-mode tensorrt

# Run only YOLO PyTorch path (no TensorRT YOLO comparison)
bash scripts/run_all_benchmarks.sh --yolo-mode pytorch

# Skip preparation (if benchmark/images and benchmark/videos are already ready)
bash scripts/run_all_benchmarks.sh --skip-prepare
```

By default, the script runs:

- yolo26 (x, pytorch)
- yolo26 (n, pytorch)
- yolo26 (x, tensorrt)
- yolo26 (n, tensorrt)
- sam21
- sam3
- fastsam
- mobilesam

Jetson-first defaults are enabled out of the box:

- `--hardware jetson`
- `--yolo-mode compare` (runs YOLO pytorch + tensorrt side by side)
- `--video-capture auto` and `--video-writer auto` (Jetson uses GStreamer when available)
- output saving disabled by default for best throughput

For laptop/desktop runs, override as needed, e.g.:

```bash
bash scripts/run_all_benchmarks.sh --hardware desktop --yolo-mode pytorch --save-outputs

# Keep side-by-side YOLO comparison, but save outputs
bash scripts/run_all_benchmarks.sh --save-outputs
```

The script uses `benchmark.py all` so outputs and metrics remain end-to-end (inference + overlay + write).
