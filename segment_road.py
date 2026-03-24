#!/usr/bin/env python3
"""
Off-Road Traversable Road Segmentation
=======================================
Segment the traversable road in off-road environments using either:
  - SAM3   — Meta's Segment Anything Model 3 (open-vocabulary concept segmentation)
  - YOLO26 — Ultralytics YOLOE-26 (open-vocabulary instance segmentation)

Both backends accept the same text prompts and report identical metrics so you
can make a direct, apples-to-apples comparison.

Usage
-----
  python segment_road.py --model yolo26 --input test_data/
  python segment_road.py --model sam3   --input test_data/

  python segment_road.py --model yolo26 --input test_data/orfd.png
  python segment_road.py --model yolo26 --input test_data/aa.mp4 --model-size s

  # Custom prompts
  python segment_road.py --model yolo26 --prompts "dirt trail" "mud road" "gravel path"

  # Save JSON metrics report
  python segment_road.py --model yolo26 --report

Notes
-----
- YOLO26 weights are downloaded automatically on first run.
- SAM3 weights (sam3.pt, ~3.4 GB) must be requested from HuggingFace:
    https://huggingface.co/facebook/sam3
  Place the downloaded sam3.pt in the working directory (or use --sam3-weights).
- If SAM3 raises "TypeError: 'SimpleTokenizer' object is not callable":
    pip uninstall clip -y
    pip install git+https://github.com/ultralytics/CLIP.git
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTS  = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

DEFAULT_PROMPTS = [
    "traversable road",
    "dirt road",
    "off-road trail",
    "drivable terrain",
    "gravel path",
]

MASK_BGR = (34, 139, 34)
CONTOUR_BGR = (0, 255, 255)
MASK_ALPHA = 0.40
HUD_ALPHA = 0.55
JETSON_MODEL_HINTS = ("jetson", "orin", "nvidia")


def _read_text_if_exists(p: Path):
    try:
        if p.exists():
            return p.read_text(errors="ignore").strip()
    except Exception:
        return None
    return None


def detect_jetson() -> bool:
    """Best-effort runtime detection for NVIDIA Jetson devices."""
    model = _read_text_if_exists(Path("/proc/device-tree/model"))
    if model and any(h in model.lower() for h in JETSON_MODEL_HINTS):
        return True
    release = _read_text_if_exists(Path("/etc/nv_tegra_release"))
    if release:
        return True
    return False


def build_video_writer(out_path: Path, fps: float, size: tuple[int, int], backend: str = "auto"):
    """Create a VideoWriter. Prefers Jetson NVENC via GStreamer when requested."""
    w, h = size

    use_gstreamer = backend == "gstreamer" or (backend == "auto" and detect_jetson())
    if use_gstreamer:
        location = str(out_path.resolve()).replace('"', '\\"')
        pipeline = (
            "appsrc ! videoconvert ! video/x-raw,format=BGR ! "
            "nvvidconv ! nvv4l2h264enc preset-level=1 insert-sps-pps=true idrinterval=30 bitrate=8000000 ! "
            f"h264parse ! qtmux ! filesink location=\"{location}\" sync=false"
        )
        gst_writer = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, fps, (w, h))
        if gst_writer.isOpened():
            return gst_writer, "gstreamer"
        print("  [WARN] GStreamer/NVENC writer unavailable. Falling back to OpenCV mp4v writer.")

    opencv_writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    return opencv_writer, "opencv"


def build_video_capture(video_path: Path, backend: str = "auto"):
    """Create a VideoCapture. Prefers Jetson NVDEC via GStreamer when requested."""
    use_gstreamer = backend == "gstreamer" or (backend == "auto" and detect_jetson())
    if use_gstreamer:
        location = str(video_path.resolve()).replace('"', '\\"')
        pipeline = (
            f"filesrc location=\"{location}\" ! "
            "qtdemux ! h264parse ! nvv4l2decoder ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true sync=false"
        )
        gst_cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if gst_cap.isOpened():
            return gst_cap, "gstreamer"
        print("  [WARN] GStreamer/NVDEC capture unavailable. Falling back to OpenCV capture.")

    opencv_cap = cv2.VideoCapture(str(video_path))
    return opencv_cap, "opencv"

def collect_inputs(input_path: str):
    """
    Return (image_paths, video_paths, label_dir) for the given file or directory.

    Handles two layouts:
      flat/          →  test_data/orfd.png  (no labels)
      structured/    →  test_data/raw/orfd.png  +  test_data/labeled/orfd_labeled.png
    """
    p = Path(input_path)

    if p.is_file():
        ext = p.suffix.lower()

        if ext in IMAGE_EXTS:
            return [p], [], None

        if ext in VIDEO_EXTS:
            return [], [p], None

        sys.exit(f"[ERROR] Unsupported file type: {ext}")

    if p.is_dir():
        raw_dir = p / "raw"
        label_dir = p / "labeled"
        src_dir = raw_dir if raw_dir.is_dir() else p
        ldir = label_dir if label_dir.is_dir() else None
        imgs = sorted(f for f in src_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        vids = sorted(f for f in src_dir.iterdir() if f.suffix.lower() in VIDEO_EXTS)

        return imgs, vids, ldir

    sys.exit(f"[ERROR] Path not found: {input_path}")


def find_label(img_path: Path, label_dir: Path):
    """Return the label path for an image, or None if not found.
    Convention: raw/orfd.png → labeled/orfd_labeled.png
    """
    candidate = label_dir / f"{img_path.stem}_labeled{img_path.suffix}"

    return candidate if candidate.exists() else None


def load_gt_mask(label_path: Path):
    """
    Load a 3-class label image and return binary masks.

    Label pixel values:
      255 (white) = traversable road  → positive class
        0 (black) = non-road          → negative class
      128 (gray)  = void / unknown    → excluded from all metrics

    Returns
    -------
    gt_mask   : np.ndarray [H, W] bool   True = road
    void_mask : np.ndarray [H, W] bool   True = ignore this pixel
    """
    label = cv2.imread(str(label_path))

    if label is None:
        return None, None

    gray = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY) if label.ndim == 3 else label
    gt_mask = gray > 200
    void_mask = (gray > 50) & (gray < 200)

    return gt_mask, void_mask


def compute_gt_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, void_mask: np.ndarray) -> dict:
    """
    Compute segmentation accuracy against ground truth, excluding void pixels.

    Metrics
    -------
    iou            Intersection over Union  = TP / (TP + FP + FN)
    precision      TP / (TP + FP)
    recall         TP / (TP + FN)
    f1             2 * precision * recall / (precision + recall)
    pixel_accuracy (TP + TN) / all valid pixels
    """
    pred_mask = np.squeeze(pred_mask)
    gt_mask = np.squeeze(gt_mask)
    void_mask = np.squeeze(void_mask)

    valid = ~void_mask
    pred_v = pred_mask[valid]
    gt_v = gt_mask[valid]

    TP = int(( pred_v &  gt_v).sum())
    TN = int((~pred_v & ~gt_v).sum())
    FP = int(( pred_v & ~gt_v).sum())
    FN = int((~pred_v &  gt_v).sum())

    den = TP + FP + FN
    if den > 0:
        iou = TP / den
    else:
        iou = 0.0

    den = TP + FP
    if den > 0:
        precision = TP / den
    else:
        precision = 0.0

    den = TP + FN
    if den > 0:
        recall = TP / den
    else:
        recall = 0.0

    den = precision + recall
    if den > 0:
        f1 = 2 * precision * recall / den
    else:
        f1 = 0.0

    den = TP + TN + FP + FN
    if den > 0:
        pixel_acc = (TP + TN) / den
    else:
        pixel_acc = 0.0

    return {
        "iou": round(iou, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "pixel_accuracy": round(pixel_acc, 4),
    }


def extract_masks(result, h: int, w: int):
    """
    Extract and combine all segmentation masks from an Ultralytics result object.

    Returns
    -------
    combined : np.ndarray [H, W] bool
        Union of all detected road instance masks.
    confs : list[float]
        Per-instance confidence scores.
    """
    combined = np.zeros((h, w), dtype=bool)
    confs = []

    if result is None or result.masks is None:
        return combined, confs

    for i, m in enumerate(result.masks.data):
        m_np = m.cpu().numpy().astype(np.uint8)

        if m_np.shape != (h, w):
            m_np = cv2.resize(m_np, (w, h), interpolation=cv2.INTER_NEAREST)

        combined |= m_np.astype(bool)
        conf = 1.0

        if result.boxes is not None and i < len(result.boxes.conf):
            conf = float(result.boxes.conf[i].cpu())
        elif (hasattr(result.masks, "conf") and result.masks.conf is not None and i < len(result.masks.conf)):
            conf = float(result.masks.conf[i].cpu())

        confs.append(conf)

    return combined, confs


def overlay_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply semi-transparent green overlay and cyan contour boundary."""
    if not mask.any():
        return frame.copy()

    out = frame.copy()
    mask_u8 = mask.astype(np.uint8)
    x, y, w, h = cv2.boundingRect(mask_u8)
    if w > 0 and h > 0:
        roi = out[y : y + h, x : x + w]
        roi_mask = mask[y : y + h, x : x + w]
        roi_overlay = roi.copy()
        roi_overlay[roi_mask] = MASK_BGR
        roi[:] = cv2.addWeighted(roi_overlay, MASK_ALPHA, roi, 1.0 - MASK_ALPHA, 0)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, CONTOUR_BGR, 2)

    return out


def draw_hud(frame: np.ndarray, metrics: dict, model_label: str) -> np.ndarray:
    """Render a semi-transparent HUD with per-frame metrics in the top-left corner."""
    lines = [
        f"Model : {model_label}",
        f"Infer : {metrics['inference_time_ms']:.0f} ms   FPS: {metrics['fps']:.1f}",
        f"Cover : {metrics['road_coverage_pct']:.1f}%   Det: {metrics['num_detections']}",
        f"Conf  : {metrics['mean_confidence']:.3f}",
    ]

    if "temporal_iou" in metrics:
        lines.append(f"T-IoU : {metrics['temporal_iou']:.3f}")

    if "iou" in metrics:
        lines.append(f"IoU   : {metrics['iou']:.3f}   F1: {metrics['f1']:.3f}")
        lines.append(f"Prec  : {metrics['precision']:.3f}   Rec: {metrics['recall']:.3f}")

    pad, line_h = 8, 24
    rect_h = len(lines) * line_h + pad * 2
    rect_w = 370

    h, w = frame.shape[:2]
    x2 = min(rect_w, w)
    y2 = min(rect_h, h)
    if x2 > 0 and y2 > 0:
        roi = frame[:y2, :x2]
        overlay = roi.copy()
        cv2.rectangle(overlay, (0, 0), (x2, y2), (0, 0, 0), -1)
        roi[:] = cv2.addWeighted(overlay, HUD_ALPHA, roi, 1.0 - HUD_ALPHA, 0)

    for i, ln in enumerate(lines):
        y = pad + (i + 1) * line_h - 4
        cv2.putText(frame, ln, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 255, 180), 1, cv2.LINE_AA,)

    return frame


def compute_metrics(
    mask: np.ndarray,
    confs: list,
    t_ms: float,
    prev_mask=None,
) -> dict:
    """
    Compute per-frame / per-image metrics.

    Metrics
    -------
    inference_time_ms : float   Wall-clock inference time in milliseconds.
    fps               : float   Effective frames per second from inference time.
    road_coverage_pct : float   Fraction of image covered by road mask (%).
    mean_confidence   : float   Mean confidence across all detected instances.
    num_detections    : int     Number of distinct road instances found.
    temporal_iou      : float   IoU with the previous frame's mask (video only).
    """
    h, w = mask.shape
    road_px = int(mask.sum())

    m = {
        "inference_time_ms": round(t_ms, 2),
        "fps": round(1000.0 / t_ms, 1) if t_ms > 0 else 0.0,
        "road_coverage_pct": round(100.0 * road_px / (h * w), 2),
        "mean_confidence": round(float(np.mean(confs)), 4) if confs else 0.0,
        "num_detections": len(confs),
    }

    if prev_mask is not None:
        inter = int((mask & prev_mask).sum())
        union = int((mask | prev_mask).sum())
        m["temporal_iou"] = round(inter / union, 4) if union > 0 else 1.0

    return m


def print_summary(all_m: list, model_name: str, label: str):
    """Print a formatted aggregate-metrics table to stdout."""
    if not all_m:
        return

    col_w = 30
    sep  = "=" * 66
    dash = "-" * 66

    def _section(title, keys):
        vals_exist = any(k in all_m[0] for k in keys)

        if not vals_exist:
            return

        print(f"  -- {title} --")

        for k in keys:
            vals = [m[k] for m in all_m if k in m]

            if vals:
                print(f"  {k:{col_w}}  {np.mean(vals):>8.3f}  {np.min(vals):>8.3f}  {np.max(vals):>8.3f}")

    print(f"\n{sep}")
    print(f"  Results: {model_name.upper():<8}  |  {label}")
    print(dash)
    print(f"  {'Metric':{col_w}}  {'Mean':>8}  {'Min':>8}  {'Max':>8}")
    print(dash)

    _section("Performance", ["inference_time_ms", "fps"])
    _section("Model output", ["road_coverage_pct", "mean_confidence", "num_detections"])
    _section("Video consistency", ["temporal_iou"])
    _section("Ground truth", ["iou", "f1", "precision", "recall", "pixel_accuracy"])

    print(f"{sep}\n")

class YOLO26Segmentor:
    """
    YOLOE-26 open-vocabulary segmentor.

    Uses Ultralytics YOLOE-26-seg with text prompts, so it works on arbitrary
    off-road road concepts without any fine-tuning.
    """
    def __init__(
        self,
        size: str = "x",
        conf: float = 0.25,
        imgsz: int = 640,
        prompts: list = None,
        weights: str | None = None,
        backend: str = "pytorch",
        engine: str | None = None,
    ):
        from ultralytics import YOLO

        name = weights or f"yoloe-26{size}-seg.pt"
        backend = backend.lower().strip()

        self.conf = conf
        self.imgsz = imgsz
        self.prompts = prompts or DEFAULT_PROMPTS

        if backend == "tensorrt":
            print(f"[YOLO26] Loading {name} for TensorRT export ...")
            base = YOLO(name)

            # CRITICAL: Set text classes BEFORE exporting to TensorRT
            # This bakes the text embeddings into the engine
            print(f"[YOLO26] Setting text classes: {self.prompts}")
            base.set_classes(self.prompts)

            engine_path = Path(engine) if engine else Path(str(name)).with_suffix(".engine")
            if engine_path.exists():
                print(f"[YOLO26] Loading existing TensorRT engine {engine_path} ...")
                print(f"[YOLO26] [WARN] Using existing engine - if text classes changed, delete {engine_path} to re-export")
                self.model = YOLO(str(engine_path))
            else:
                print(
                    f"[YOLO26] Exporting TensorRT engine from {name} "
                    f"(imgsz={imgsz}, half=True, first run may take time) ..."
                )
                exported = base.export(format="engine", half=True, imgsz=imgsz)
                exported_path = Path(str(exported))
                print(f"[YOLO26] TensorRT engine exported: {exported_path}")
                print(f"[YOLO26] Loading TensorRT engine {exported_path} ...")
                self.model = YOLO(str(exported_path))
        else:
            print(f"[YOLO26] Loading {name} (auto-download if not cached) ...")
            self.model = YOLO(name)
            try:
                self.model.set_classes(self.prompts)
            except Exception as e:
                print(f"[YOLO26] [WARN] Could not set text classes on current backend: {e}")
        # YOLOE with text prompts uses a CLIP text encoder that stays in float32,
        # so half=True causes a dtype mismatch. Keep full precision.
        print(f"[YOLO26] Text classes : {self.prompts}")
        print(f"[YOLO26] Inference image size: {self.imgsz}")

    def infer(self, source) -> tuple:
        """Run inference on a file path or numpy BGR frame."""
        t0  = time.perf_counter()
        res = self.model.predict(source, conf=self.conf, imgsz=self.imgsz, verbose=False)
        t_ms = (time.perf_counter() - t0) * 1000

        return res[0], t_ms


class SAM3Segmentor:
    """
    SAM3 semantic predictor for concept segmentation.

    Uses SAM3SemanticPredictor for images (text prompt) and
    SAM3VideoSemanticPredictor for videos (temporal tracking).
    """
    def __init__(self, weights: str = "sam3.pt", conf: float = 0.25, prompts: list = None):
        weights_path = Path(weights)

        if not weights_path.exists():
            sys.exit(
                f"\n[ERROR] SAM3 weights not found: {weights}\n"
                "  1. Request access at: https://huggingface.co/facebook/sam3\n"
                "  2. Download sam3.pt and place it in the working directory.\n"
                "  3. Or pass the path via --sam3-weights /path/to/sam3.pt\n"
            )

        from ultralytics.models.sam import SAM3SemanticPredictor

        print(f"[SAM3] Loading {weights} ...")
        self._ov = dict(
            conf=conf,
            task="segment",
            mode="predict",
            model=str(weights_path),
            verbose=False,
            save=False,       # prevent writing result images to disk on every call
            save_txt=False,
            save_conf=False,
        )
        self.predictor = SAM3SemanticPredictor(overrides=self._ov)
        self.prompts = prompts or DEFAULT_PROMPTS
        self.weights = str(weights_path)
        self.conf = conf
        print(f"[SAM3] Text prompts : {self.prompts}")

    def infer(self, source) -> tuple:
        """Run image inference. source = file path string or numpy BGR frame."""
        t0 = time.perf_counter()
        self.predictor.set_image(source)
        res  = self.predictor(text=self.prompts)
        t_ms = (time.perf_counter() - t0) * 1000

        return (res[0] if res else None), t_ms


class SAM21Segmentor:
    """
    SAM 2.1 Large segmentor using a fixed point prompt.

    SAM 2.1 does not support text prompts, so a single foreground point is
    placed at the bottom-centre of the frame (x=50%, y=75%) — where the road
    consistently appears in a forward-facing camera.  The model is downloaded
    automatically by Ultralytics on first use (~224 MB).
    """

    WEIGHTS = "sam2.1_l.pt"
    # Relative position of the road prompt point (fraction of frame dims)
    POINT_X = 0.50
    POINT_Y = 0.75

    def __init__(self, weights: str = WEIGHTS, conf: float = 0.25):
        from ultralytics import SAM

        print(f"[SAM21] Loading {weights} (auto-download if not cached) ...")
        self.model = SAM(weights)
        self.conf  = conf
        print(f"[SAM21] Point prompt: centre-x={self.POINT_X:.0%}  y={self.POINT_Y:.0%} of frame")

    def infer(self, source) -> tuple:
        """
        Run inference on a file path or numpy BGR frame.

        Converts file paths to numpy arrays internally so image dimensions are
        always available for computing the point prompt without a second I/O hit.
        """
        if isinstance(source, np.ndarray):
            frame = source
        else:
            frame = cv2.imread(str(source))
            if frame is None:
                return None, 0.0

        h, w = frame.shape[:2]
        cx = int(w * self.POINT_X)
        cy = int(h * self.POINT_Y)

        t0  = time.perf_counter()
        res = self.model.predict(
            frame,
            points=[[cx, cy]],
            labels=[1],          # 1 = foreground
            conf=self.conf,
            verbose=False,
        )
        t_ms = (time.perf_counter() - t0) * 1000

        return (res[0] if res else None), t_ms


def run_image(seg, img_path: Path, out_dir: Path, model_name: str, label_dir=None, return_mask: bool = False):
    """Segment a single image. Saves annotated output, returns metrics dict (and optionally mask)."""
    img = cv2.imread(str(img_path))

    if img is None:
        print(f"  [WARN] Cannot read: {img_path.name}")
        return None

    result, t_ms = seg.infer(str(img_path))
    h, w = img.shape[:2]
    mask, confs = extract_masks(result, h, w)

    m = compute_metrics(mask, confs, t_ms)
    m["file"] = img_path.name

    if label_dir is not None:
        lp = find_label(img_path, label_dir)

        if lp:
            gt_mask, void_mask = load_gt_mask(lp)

            if gt_mask is not None:
                m.update(compute_gt_metrics(mask, gt_mask, void_mask))
        else:
            print(f"  [WARN] No label found for {img_path.name} in {label_dir}")

    annotated = overlay_mask(img, mask)
    annotated = draw_hud(annotated, m, model_name.upper())

    out = out_dir / f"{img_path.stem}_{model_name}_road{img_path.suffix}"
    cv2.imwrite(str(out), annotated)

    gt_str = (f"  IoU={m['iou']:.3f}  F1={m['f1']:.3f}"
               if "iou" in m else "  (no label)")
    print(
        f"  {img_path.name:30s}  "
        f"cov={m['road_coverage_pct']:5.1f}%  "
        f"t={t_ms:.0f}ms"
        f"{gt_str}  → {out.name}"
    )

    if return_mask:
        return m, mask

    return m


def run_video(
    seg,
    vid_path: Path,
    out_dir: Path,
    model_name: str,
    video_writer: str = "auto",
    video_capture: str = "auto",
) -> list:
    """
    Segment a video. Saves annotated MP4 output.
    Returns a list of per-frame metrics dicts.
    """
    probe, capture_backend = build_video_capture(vid_path, backend=video_capture)

    if not probe.isOpened():
        print(f"  [WARN] Cannot open: {vid_path.name}")
        return []

    src_fps  = probe.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    probe.release()

    out_path = out_dir / f"{vid_path.stem}_{model_name}_road.mp4"
    writer, writer_backend = build_video_writer(out_path, src_fps, (W, H), backend=video_writer)
    print(
        f"\n  [{model_name.upper()}] {vid_path.name}  "
        f"({W}×{H}  {src_fps:.1f} fps  {n_frames} frames)  "
        f"capture={capture_backend}  writer={writer_backend}"
    )

    if not writer.isOpened():
        print(f"  [WARN] Cannot open output writer for: {out_path.name}")
        return []

    all_m, prev_mask, fi = [], None, 0
    cap, _ = build_video_capture(vid_path, backend=video_capture)

    if not cap.isOpened():
        print(f"  [WARN] Cannot open input capture for: {vid_path.name}")
        writer.release()
        return []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        result, t_ms = seg.infer(frame)
        mask, confs  = extract_masks(result, H, W)
        m = compute_metrics(mask, confs, t_ms, prev_mask)
        m["frame"] = fi

        annotated = overlay_mask(frame, mask)
        annotated = draw_hud(annotated, m, model_name.upper())
        writer.write(annotated)
        all_m.append(m)
        prev_mask = mask
        fi += 1

        if fi % 30 == 0:
            print(
                f"    frame {fi:4d}/{n_frames}  "
                f"cov={m['road_coverage_pct']:.1f}%  "
                f"conf={m['mean_confidence']:.3f}  "
                f"t={t_ms:.0f}ms"
            )

    cap.release()
    writer.release()
    print(f"  Saved: {out_path.name}  ({fi} frames processed)")

    return all_m

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Off-road traversable road segmentation — SAM3 vs YOLO26",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model", required=True, choices=["sam3", "sam21", "yolo26"],
        help="Segmentation backend: 'sam3', 'sam21', or 'yolo26'",
    )
    p.add_argument(
        "--input", default="test_data",
        help="Image/video file or directory to process  [default: test_data/]",
    )
    p.add_argument(
        "--output", default="output",
        help="Directory for annotated outputs           [default: output/]",
    )
    p.add_argument(
        "--conf", type=float, default=0.25,
        help="Detection confidence threshold            [default: 0.25]",
    )
    p.add_argument(
        "--prompts", nargs="+", default=None,
        help="Custom text prompts for the road concept  [default: built-in set]",
    )
    p.add_argument(
        "--model-size", default="x", choices=["n", "s", "m", "l", "x"],
        help="YOLOE-26 model size variant               [default: x, ignored for sam3]",
    )
    p.add_argument(
        "--imgsz", type=int, default=640,
        help="YOLO inference/export image size (pixels) [default: 640, YOLO only]",
    )
    p.add_argument(
        "--yolo-weights", default=None,
        help="Path to local YOLO weights/engine (optional).",
    )
    p.add_argument(
        "--yolo-backend",
        default="pytorch",
        choices=["pytorch", "tensorrt"],
        help="YOLO inference backend: pytorch or tensorrt.",
    )
    p.add_argument(
        "--yolo-engine", default=None,
        help="Path to TensorRT .engine file (used when --yolo-backend tensorrt).",
    )
    p.add_argument(
        "--sam3-weights", default="sam3.pt",
        help="Path to sam3.pt weights file              [default: sam3.pt]",
    )
    p.add_argument(
        "--sam21-weights", default="sam2.1_l.pt",
        help="Path to SAM 2.1 weights (auto-downloaded) [default: sam2.1_l.pt]",
    )
    p.add_argument(
        "--video-writer",
        default="auto",
        choices=["auto", "opencv", "gstreamer"],
        help="Video writer backend: auto (Jetson->GStreamer), opencv, or gstreamer.",
    )
    p.add_argument(
        "--video-capture",
        default="auto",
        choices=["auto", "opencv", "gstreamer"],
        help="Video capture backend: auto (Jetson->GStreamer), opencv, or gstreamer.",
    )
    p.add_argument(
        "--report", action="store_true",
        help="Save a metrics_<model>.json report to the output directory",
    )
    return p


def main():
    args = build_parser().parse_args()

    out_dir = Path(args.output) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    images, videos, label_dir = collect_inputs(args.input)
    total = len(images) + len(videos)

    hdr = "-" * 66
    print(f"\n{hdr}")
    print(f"  Model   : {args.model.upper()}")
    print(f"  Inputs  : {len(images)} image(s), {len(videos)} video(s)  [{args.input}]")
    print(f"  Labels  : {label_dir or 'none found'}")
    print(f"  Output  : {out_dir.resolve()}")
    if args.model != "sam21":
        print(f"  Prompts : {args.prompts or DEFAULT_PROMPTS}")
    print(hdr)

    if total == 0:
        sys.exit("[ERROR] No supported image/video files found in the input path.")

    if args.model == "yolo26":
        seg = YOLO26Segmentor(
            size=args.model_size,
            conf=args.conf,
            imgsz=args.imgsz,
            prompts=args.prompts,
            weights=args.yolo_weights,
            backend=args.yolo_backend,
            engine=args.yolo_engine,
        )
    elif args.model == "sam21":
        seg = SAM21Segmentor(weights=args.sam21_weights, conf=args.conf)
    else:
        seg = SAM3Segmentor(weights=args.sam3_weights, conf=args.conf, prompts=args.prompts)

    report = {"model": args.model, "images": [], "videos": {}}

    if images:
        print(f"\n[Images]")
        img_metrics = []

        for p in images:
            m = run_image(seg, p, out_dir, args.model, label_dir)

            if m:
                img_metrics.append(m)

        report["images"] = img_metrics

        if img_metrics:
            print_summary(img_metrics, args.model, f"{len(img_metrics)} image(s)")

    if videos:
        print(f"\n[Videos]")

        for p in videos:
            vm = run_video(
                seg,
                p,
                out_dir,
                args.model,
                video_writer=args.video_writer,
                video_capture=args.video_capture,
            )
            report["videos"][p.name] = vm

            if vm:
                print_summary(vm, args.model, p.name)

    if args.report:
        rp = out_dir / f"metrics_{args.model}.json"
        with open(rp, "w") as f:
            json.dump(report, f, indent=2)
        print(f"JSON metrics report → {rp}")

    print(f"\nAll outputs saved to: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
