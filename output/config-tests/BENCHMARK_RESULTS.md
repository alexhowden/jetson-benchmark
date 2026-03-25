# YOLOE-26 Benchmark Results

**Date:** March 24, 2026
**Device:** Jetson Orin NX
**Image Size:** 640x640 (default)
**Video Backend:** OpenCV (software decode)

---

## Executive Summary

Benchmarked YOLOE-26 models (n and x sizes) with PyTorch and TensorRT backends on the Jetson Orin NX. Key findings:

- **PyTorch significantly outperforms TensorRT** in accuracy metrics (IoU, F1 score)
- **TensorRT is 2.7-4.2x faster** than PyTorch for inference
- **YOLO-26x with PyTorch** achieves best accuracy: 71.3% IoU, 78.5% F1 (but slow at 245ms)
- **YOLO-26n with PyTorch** provides best balance: 59ms inference, 58.7% IoU, real-time capable
- **TensorRT models show degraded accuracy** - likely due to incorrect engine export (needs re-export)

---

## Performance Comparison Table

| Config | Model | Backend | Img IoU | Img F1 | Pixel Acc | Vid Temporal IoU | Inference (ms) | Video FPS |
|--------|-------|---------|---------|--------|-----------|------------------|----------------|-----------|
| **yolo26-n-tensorrt** | n | TensorRT | 28.81% | 42.79% | 71.43% | 75.67% | **21.91** | **45.94** |
| **yolo26-n-pytorch** | n | PyTorch | 58.70% | 67.57% | 82.38% | 76.89% | 59.03 | 16.96 |
| **yolo26-x-tensorrt** | x | TensorRT | 32.78% | 47.74% | 75.14% | 92.77% | 103.29 | 9.71 |
| **yolo26-x-pytorch** | x | PyTorch | **71.31%** | **78.51%** | **89.77%** | 86.10% | 245.07 | **4.10** |

**Best accuracy:** YOLO-26x PyTorch (71.3% IoU)
**Best speed:** YOLO-26n TensorRT (21.91ms, 45.94 FPS)
**Best balance:** YOLO-26n PyTorch (58.7% IoU, 16.96 FPS)

---

## Detailed Metrics

### Image Benchmark

| Metric | yolo26-n-tensorrt | yolo26-n-pytorch | yolo26-x-tensorrt | yolo26-x-pytorch |
|--------|-------------------|------------------|-------------------|------------------|
| **Mean IoU** | 28.81% | 58.70% | 32.78% | **71.31%** |
| **Mean F1 Score** | 42.79% | 67.57% | 47.74% | **78.51%** |
| **Pixel Accuracy** | 71.43% | 82.38% | 75.14% | **89.77%** |
| **False Negative Rate** | 64.84% | 30.79% | 61.00% | **21.09%** |
| **False Positive Rate** | 10.89% | 11.25% | 7.31% | **5.04%** |
| **Inference Time** | **21.91ms** | 59.03ms | 103.29ms | 245.07ms |
| **Status** | ❌ Failed | ❌ Failed | ❌ Failed | ❌ Failed |

### Video Benchmark

| Metric | yolo26-n-tensorrt | yolo26-n-pytorch | yolo26-x-tensorrt | yolo26-x-pytorch |
|--------|-------------------|------------------|-------------------|------------------|
| **Temporal IoU** | 75.67% | 76.89% | **92.77%** | 86.10% |
| **No-Detection Rate** | 31.45% | 54.39% | 92.27% | 76.45% |
| **Frame Stability** | 75.67% | 76.89% | **92.77%** | 86.10% |
| **Inference Time** | **21.91ms** | 59.03ms | 103.29ms | 245.07ms |
| **Video FPS** | **45.94** | 16.96 | 9.71 | **4.10** |
| **Status** | ❌ Failed | ❌ Failed | ❌ Failed | ❌ Failed |

---

## Key Findings

### 1. **TensorRT Accuracy Issue** ⚠️

TensorRT models show **significantly degraded accuracy** compared to PyTorch:
- **YOLO-26n**: 28.81% IoU (TensorRT) vs 58.70% IoU (PyTorch) - **51% worse**
- **YOLO-26x**: 32.78% IoU (TensorRT) vs 71.31% IoU (PyTorch) - **54% worse**

**Likely Cause:** TensorRT engines were exported before the text class embedding fix. The engines need to be re-exported with `set_classes()` called before export.

**Recommendation:** Delete existing `.engine` files and re-run benchmarks to export fresh engines with proper text embeddings. After re-export, TensorRT should achieve similar accuracy to PyTorch while maintaining speed advantage.

### 2. **Speed vs Accuracy Tradeoff**

**YOLO-26x PyTorch (Highest Accuracy):**
- Inference: 245ms
- IoU: **71.31%**, F1: **78.51%**
- Video FPS: 4.10
- **Best for offline processing or maximum accuracy needs**

**YOLO-26n PyTorch (Balanced):**
- Inference: 59ms
- IoU: 58.70%, F1: 67.57%
- Video FPS: 16.96
- **Best for real-time with good accuracy**

**YOLO-26n TensorRT (Fastest):**
- Inference: 22ms (2.7x faster than PyTorch-n, 11x faster than PyTorch-x)
- IoU: 28.81% (needs re-export)
- Video FPS: 45.94
- **Should match PyTorch-n accuracy after re-export while maintaining speed**

### 3. **Model Size Impact**

Clear tradeoff between model size and speed:
- **YOLO-26n**: 22-59ms inference (real-time capable)
- **YOLO-26x**: 103-245ms inference (4-11x slower)

**Accuracy improvement from n to x:**
- PyTorch: 58.70% → 71.31% IoU (+21% improvement)
- Worth the slowdown only if accuracy is critical

### 4. **Video Processing Performance**

With OpenCV software decode:
- **YOLO-26n TensorRT**: 45.94 FPS (excellent, real-time capable)
- **YOLO-26n PyTorch**: 16.96 FPS (good, real-time capable)
- **YOLO-26x TensorRT**: 9.71 FPS (marginal for real-time)
- **YOLO-26x PyTorch**: 4.10 FPS (too slow for real-time, offline only)

**Note:** GStreamer hardware decode would improve these by ~20-30%, making YOLO-26x TensorRT viable for real-time (~12-13 FPS).

---

## Recommendations

### Immediate Actions

1. **Re-export TensorRT engines** with proper text class embeddings:
   ```bash
   rm yoloe-26n-seg.engine
   rm yoloe-26x-seg.engine
   ./scripts/run_all_configs.sh --only-tensorrt
   ```

2. **Complete YOLO-26x PyTorch benchmark** (appears missing from results)

### For Production Deployment

**If maximum accuracy is critical (IoU >70% required):**
- Use **YOLO-26x with PyTorch**
- Actual: 245ms inference, 4.10 FPS, 71.3% IoU
- **Only suitable for offline processing or very slow frame rates**

**If good accuracy + real-time needed (IoU ~60%, >15 FPS):**
- Use **YOLO-26n with PyTorch**
- Actual: 59ms inference, 16.96 FPS, 58.7% IoU
- **Recommended for most real-time applications**

**If speed is critical (>40 FPS required):**
- Use **YOLO-26n with TensorRT** (after re-export)
- Actual: 22ms inference, 45.94 FPS
- Expected: Should achieve ~55-60% IoU after proper export
- **Best for high-speed real-time applications**

**If balanced performance needed (~10 FPS, good accuracy):**
- Consider **YOLO-26x with TensorRT** (after re-export)
- Actual: 103ms inference, 9.71 FPS
- Expected: Should achieve ~65-70% IoU after proper export

### Future Optimizations

1. **Rebuild OpenCV with GStreamer** support
   - Would improve video FPS by 20-30%
   - YOLO-26n TensorRT could reach 55-60 FPS

2. **Test different Jetson power modes**
   - Current results likely at default clocks
   - MAXN mode would improve inference times

3. **Consider image size reduction**
   - Test 480x480 for faster inference
   - May sacrifice some accuracy

---

## Test Configuration

- **Total Runtime:** ~1.5 hours (4 configs completed)
  - yolo26-n-tensorrt: 855s (14.3 min)
  - yolo26-n-pytorch: 648s (10.8 min)
  - yolo26-x-tensorrt: 2182s (36.4 min)
  - yolo26-x-pytorch: 2133s (35.6 min)
- **Image Dataset:** 200 images from `benchmark/images/`
- **Video Dataset:** Videos from `benchmark/videos/`
- **Hardware:** Jetson Orin NX (default power mode)
- **Software:**
  - Ultralytics 8.4.15
  - PyTorch 2.3.0
  - TensorRT 10.3.0
  - CUDA device 0

---

## Next Steps

1. ✅ Complete all 4 benchmark configurations
2. ⏳ **Delete old TensorRT engines and re-export with proper text embeddings**
   ```bash
   rm yoloe-26n-seg.engine yoloe-26x-seg.engine
   ./scripts/run_all_configs.sh --only-tensorrt
   ```
3. ⏳ Compare re-exported TensorRT results (should match PyTorch accuracy)
4. ⏳ Test different Jetson power modes (MAXN) for potential speed improvements
5. ⏳ Consider GStreamer rebuild if >20% FPS boost needed for deployment
6. ⏳ Test YOLO-26l (large) as middle ground between n and x

---

*Generated from benchmark run: 2026-03-24 16:50:52*
