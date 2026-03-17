#!/usr/bin/env python3
"""
Benchmark runner for Off-Road Traversable Road Segmentation.

Creates a fixed benchmark set (200 images + 10 videos) and runs a chosen model
backend (SAM3 / SAM2.1 / YOLOE-26) to produce metrics.

Folders
-------
benchmark/
  images/
    raw/       (benchmark images)
    labeled/   (matching *_labeled masks)
  videos/      (put videos here manually for now)

Usage
-----
Prepare 200 labeled images from test_data/:
  python benchmark.py prepare --n-images 200

Run the benchmark on the prepared images:
  python benchmark.py run --model sam3
  python benchmark.py run --model sam2
  python benchmark.py run --model yoloe26 --model-size x
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

import segment_road as sr


BENCH_ROOT = Path("benchmark")
BENCH_IMAGES = BENCH_ROOT / "images"
BENCH_IMAGES_RAW = BENCH_IMAGES / "raw"
BENCH_IMAGES_LABELED = BENCH_IMAGES / "labeled"
BENCH_VIDEOS = BENCH_ROOT / "videos"

JETSON_MODEL_HINTS = ("jetson", "orin", "nvidia")


IMAGE_THRESHOLDS = {
    "iou": 0.85,              # mIoU / IoU >= 0.85
    "f1": 0.90,               # F1 >= 0.90
    "pixel_accuracy": 0.90,   # pixel accuracy >= 0.90
    "false_negative_rate": 0.08,  # <= 0.08
    "false_positive_rate": 0.08,  # <= 0.08
}

VIDEO_THRESHOLDS = {
    "temporal_iou": 0.85,          # >= 0.85
    "no_detection_rate": 0.01,     # <= 0.01
    "frame_to_frame_stability": 0.90,  # >= 0.90 (alias of temporal IoU here)
}

JETSON_THRESHOLDS = {
    "gpu_utilization_pct": 95.0,     # <= 95%
    "cpu_utilization_pct": 80.0,     # <= 80%
    "ram_usage_pct": 80.0,           # <= 80%
    "image_inference_speed_ms": 200.0,  # <= 200 ms
    "video_inference_speed_ms": 25.0,   # <= 25 ms
    "video_fps": 25.0,               # >= 25 FPS
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display: str


MODEL_ALIASES = {
    # canonical -> itself
    "sam3": "sam3",
    "sam21": "sam21",
    "yolo26": "yolo26",
    # user-friendly
    "sam2": "sam21",
    "sam2.1": "sam21",
    "sam2.1l": "sam21",
    "yoloe26": "yolo26",
    "yoloe-26": "yolo26",
    "yoloe-26x": "yolo26",
    "yoloe26x": "yolo26",
}


def _ensure_dirs():
    BENCH_IMAGES_RAW.mkdir(parents=True, exist_ok=True)
    BENCH_IMAGES_LABELED.mkdir(parents=True, exist_ok=True)
    BENCH_VIDEOS.mkdir(parents=True, exist_ok=True)

def _read_text_if_exists(p: Path) -> Optional[str]:
    try:
        if p.exists():
            return p.read_text(errors="ignore").strip()
    except Exception:
        return None
    return None


def detect_jetson() -> bool:
    """
    Best-effort runtime detection for NVIDIA Jetson devices.
    No hard dependency on platform-specific packages.
    """
    model = _read_text_if_exists(Path("/proc/device-tree/model"))
    if model and any(h in model.lower() for h in JETSON_MODEL_HINTS):
        return True
    release = _read_text_if_exists(Path("/etc/nv_tegra_release"))
    if release:
        return True
    return False


def _iter_raw_images(test_data_root: Path) -> list[Path]:
    images, _, _ = sr.collect_inputs(str(test_data_root))
    return [p for p in images if p.is_file()]


def _label_for_raw(raw_img: Path, test_data_root: Path) -> Optional[Path]:
    label_dir = test_data_root / "labeled"
    if not label_dir.is_dir():
        return None
    return sr.find_label(raw_img, label_dir)


def _copy_pair(raw_img: Path, label_img: Path, dst_raw: Path, dst_labeled: Path) -> None:
    dst_raw.parent.mkdir(parents=True, exist_ok=True)
    dst_labeled.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_img, dst_raw)
    shutil.copy2(label_img, dst_labeled)

def _collect_orfd_pairs(orfd_root: Path) -> list[tuple[Path, Path]]:
    """
    ORFD dataset layout (this repo):
      datasets/ORFD/{training,validation,testing}/image_data/<id>.png
      datasets/ORFD/{training,validation,testing}/gt_image/<id>_fillcolor.png

    gt_image masks are already binary (white=road, black=non-road).
    """
    pairs: list[tuple[Path, Path]] = []
    for split in ("training", "validation", "testing"):
        split_dir = orfd_root / split
        img_dir = split_dir / "image_data"
        gt_dir = split_dir / "gt_image"
        if not img_dir.is_dir() or not gt_dir.is_dir():
            continue

        for raw in img_dir.iterdir():
            if not raw.is_file() or raw.suffix.lower() not in sr.IMAGE_EXTS:
                continue
            lab = gt_dir / f"{raw.stem}_fillcolor{raw.suffix}"
            if lab.exists():
                pairs.append((raw, lab))
    return pairs


def prepare_benchmark_images(
    *,
    source_root: Path,
    n_images: int,
    seed: int,
    clear: bool,
    source: str,
) -> dict:
    _ensure_dirs()

    if clear:
        for p in BENCH_IMAGES_RAW.glob("*"):
            if p.is_file():
                p.unlink()
        for p in BENCH_IMAGES_LABELED.glob("*"):
            if p.is_file():
                p.unlink()

    pairs: list[tuple[Path, Path]]
    src_label_note = ""

    if source == "test_data":
        raw_images = _iter_raw_images(source_root)
        pairs = []
        for raw in raw_images:
            lab = _label_for_raw(raw, source_root)
            if lab is not None and lab.exists():
                pairs.append((raw, lab))
        src_label_note = "Using test_data raw/ + labeled/ with *_labeled masks."
    elif source == "orfd":
        pairs = _collect_orfd_pairs(source_root)
        src_label_note = "Using ORFD dataset image_data/ + gt_image/*_fillcolor masks."
    else:
        raise SystemExit(f"[ERROR] Unsupported prepare source: {source!r}")

    if not pairs:
        raise SystemExit(
            f"[ERROR] No labeled image pairs found under `{source_root}` (source={source})."
        )

    rng = random.Random(seed)
    rng.shuffle(pairs)

    chosen = pairs[: min(n_images, len(pairs))]

    for raw, lab in chosen:
        # Normalize label naming to match segment_road.py convention:
        # raw/foo.png -> labeled/foo_labeled.png
        labeled_name = f"{raw.stem}_labeled{raw.suffix}"
        _copy_pair(
            raw,
            lab,
            BENCH_IMAGES_RAW / raw.name,
            BENCH_IMAGES_LABELED / labeled_name,
        )

    meta = {
        "prepared_from": str(source_root),
        "source": source,
        "source_note": src_label_note,
        "seed": seed,
        "requested_n_images": n_images,
        "available_labeled_pairs": len(pairs),
        "prepared_n_images": len(chosen),
        "raw_dir": str(BENCH_IMAGES_RAW),
        "labeled_dir": str(BENCH_IMAGES_LABELED),
        "note": "Videos are not prepared automatically. Put your 10 videos into benchmark/videos/.",
    }

    (BENCH_ROOT / "benchmark_manifest.json").write_text(json.dumps(meta, indent=2))
    return meta


def _canonical_model(model: str) -> str:
    m = model.strip().lower().replace(" ", "")
    return MODEL_ALIASES.get(m, m)


def _build_segmentor(args, canonical_model: str):
    try:
        if canonical_model == "yolo26":
            weights = getattr(args, "yolo_weights", None)
            if not weights:
                weights = f"yoloe-26{args.model_size}-seg.pt"
            wp = Path(weights)
            if not wp.exists():
                raise SystemExit(
                    "\n[ERROR] YOLOE-26 weights not found locally.\n"
                    f"  Expected: {wp}\n"
                    "  Put the `.pt` file in the project directory or pass `--yolo-weights path/to/weights.pt`.\n"
                )
            # YOLO26Segmentor expects a size, but internally it constructs the filename.
            # To force local-only weights, temporarily point CWD filename via model-size mapping.
            # If a custom path is provided, set model-size based on filename and rely on Ultralytics loading it.
            if getattr(args, "yolo_weights", None):
                # Use the explicit path by instantiating Ultralytics YOLO directly via segment_road's class logic.
                # We keep the public interface stable by swapping the expected filename into place.
                # (Ultralytics accepts paths; segment_road currently passes a name, so we override by setting size="x"
                #  and ensuring the filename we want is used.)
                # Easiest + robust: call Ultralytics here and mimic YOLO26Segmentor behavior.
                print(f"[YOLO26] Loading {wp} (local file) ...")
                seg = sr.YOLO26Segmentor(size=args.model_size, conf=args.conf, prompts=args.prompts)
                # Local import so environments without ultralytics can still use prepare, etc.
                from ultralytics import YOLO  # pyright: ignore[reportMissingImports]
                seg.model = YOLO(str(wp))
                seg.model.set_classes(seg.prompts)
                return seg
            return sr.YOLO26Segmentor(size=args.model_size, conf=args.conf, prompts=args.prompts)
        if canonical_model == "sam21":
            wp = Path(args.sam21_weights)
            if not wp.exists():
                raise SystemExit(
                    "\n[ERROR] SAM2.1 weights not found locally.\n"
                    f"  Expected: {wp}\n"
                    "  Put the `.pt` file in the project directory or pass `--sam21-weights path/to/sam2.1_l.pt`.\n"
                )
            return sr.SAM21Segmentor(weights=str(wp), conf=args.conf)
        if canonical_model == "sam3":
            wp = Path(args.sam3_weights)
            if not wp.exists():
                raise SystemExit(
                    "\n[ERROR] SAM3 weights not found locally.\n"
                    f"  Expected: {wp}\n"
                    "  Put the `.pt` file in the project directory or pass `--sam3-weights path/to/sam3.pt`.\n"
                )
            return sr.SAM3Segmentor(weights=str(wp), conf=args.conf, prompts=args.prompts)
        raise SystemExit(f"[ERROR] Unknown model: {args.model!r} (canonical: {canonical_model!r})")
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or str(e)
        raise SystemExit(
            "\n[ERROR] Missing Python dependency required by the selected model.\n"
            f"  Missing: {missing}\n\n"
            "Fix:\n"
            "  conda activate offroad-seg\n"
            "  pip install -r requirements.txt\n"
            "\n"
            "See also: README.md → 'Environment Setup (Conda)'\n"
        ) from e


def _false_rates_from_counts(TP: int, TN: int, FP: int, FN: int) -> dict:
    fn_den = TP + FN
    fp_den = TN + FP
    fnr = (FN / fn_den) if fn_den > 0 else 0.0
    fpr = (FP / fp_den) if fp_den > 0 else 0.0
    return {
        "false_negative_rate": fnr,
        "false_positive_rate": fpr,
    }


def _compute_counts(pred_mask: np.ndarray, gt_mask: np.ndarray, void_mask: np.ndarray) -> tuple[int, int, int, int]:
    pred_mask = np.squeeze(pred_mask).astype(bool)
    gt_mask = np.squeeze(gt_mask).astype(bool)
    void_mask = np.squeeze(void_mask).astype(bool)
    valid = ~void_mask
    pred_v = pred_mask[valid]
    gt_v = gt_mask[valid]
    TP = int(( pred_v &  gt_v).sum())
    TN = int((~pred_v & ~gt_v).sum())
    FP = int(( pred_v & ~gt_v).sum())
    FN = int((~pred_v &  gt_v).sum())
    return TP, TN, FP, FN


def _maybe_warmup(seg, images: list[Path], warmup: int) -> None:
    if warmup <= 0 or not images:
        return
    n = min(warmup, len(images))
    for p in images[:n]:
        try:
            seg.infer(str(p))
        except Exception:
            # Warmup is best-effort; don't fail the benchmark for it.
            return


def _run_image_metrics_only(seg, img_path: Path, label_dir: Path) -> tuple[Optional[dict], Optional[tuple[int, int, int, int]]]:
    """
    Run a single image through the model and compute metrics without writing annotated images.
    Returns (metrics_dict, optional_counts_tuple).
    """
    img = sr.cv2.imread(str(img_path))
    if img is None:
        print(f"  [WARN] Cannot read: {img_path.name}")
        return None, None

    result, t_ms = seg.infer(str(img_path))
    h, w = img.shape[:2]
    mask, confs = sr.extract_masks(result, h, w)

    m = sr.compute_metrics(mask, confs, t_ms)
    m["file"] = img_path.name

    lp = sr.find_label(img_path, label_dir)
    counts = None
    if lp and lp.exists():
        gt_mask, void_mask = sr.load_gt_mask(lp)
        if gt_mask is not None:
            m.update(sr.compute_gt_metrics(mask, gt_mask, void_mask))
            counts = _compute_counts(mask, gt_mask, void_mask)
    else:
        print(f"  [WARN] No label found for {img_path.name} in {label_dir}")

    gt_str = (f"  IoU={m['iou']:.3f}  F1={m['f1']:.3f}" if "iou" in m else "  (no label)")
    print(f"  {img_path.name:30s}  cov={m['road_coverage_pct']:5.1f}%  t={t_ms:.0f}ms{gt_str}")
    return m, counts


def _iter_benchmark_videos() -> list[Path]:
    if not BENCH_VIDEOS.is_dir():
        return []
    return sorted([p for p in BENCH_VIDEOS.iterdir() if p.is_file() and p.suffix.lower() in sr.VIDEO_EXTS])


def _run_video_metrics_only(seg, vid_path: Path) -> list[dict]:
    """
    Metrics-only video processing (no annotated MP4 output).
    Returns list of per-frame metric dicts including temporal_iou.
    """
    cap = sr.cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        print(f"  [WARN] Cannot open: {vid_path.name}")
        return []

    H = int(cap.get(sr.cv2.CAP_PROP_FRAME_HEIGHT)) or None
    W = int(cap.get(sr.cv2.CAP_PROP_FRAME_WIDTH)) or None
    n_frames = int(cap.get(sr.cv2.CAP_PROP_FRAME_COUNT)) or 0
    src_fps = float(cap.get(sr.cv2.CAP_PROP_FPS) or 0.0)

    print(f"\n  [VIDEO] {vid_path.name}  ({W}×{H}  {src_fps:.1f} fps  {n_frames} frames)")

    all_m: list[dict] = []
    prev_mask = None
    fi = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        result, _ = seg.infer(frame)
        t_ms = (time.perf_counter() - t0) * 1000.0

        if H is None or W is None:
            H, W = frame.shape[:2]
        mask, confs = sr.extract_masks(result, H, W)
        m = sr.compute_metrics(mask, confs, t_ms, prev_mask)
        m["frame"] = fi
        all_m.append(m)
        prev_mask = mask
        fi += 1

        if fi % 30 == 0:
            print(
                f"    frame {fi:4d}/{n_frames or fi}  "
                f"cov={m['road_coverage_pct']:.1f}%  "
                f"conf={m['mean_confidence']:.3f}  "
                f"t={t_ms:.0f}ms"
            )

    cap.release()
    return all_m


def run_video_benchmark(args, canonical_model: str, seg) -> dict:
    vids = _iter_benchmark_videos()
    if not vids:
        return {
            "n_videos": 0,
            "videos": {},
            "video_aggregate": None,
            "pass": None,
            "pass_details": None,
        }

    out_dir = Path(args.output) / canonical_model
    out_dir.mkdir(parents=True, exist_ok=True)

    per_video: dict[str, list[dict]] = {}
    all_frames: list[dict] = []

    for vp in vids:
        if args.save_outputs:
            vm = sr.run_video(seg, vp, out_dir, canonical_model)
        else:
            vm = _run_video_metrics_only(seg, vp)

        per_video[vp.name] = vm
        all_frames.extend(vm)

        if vm:
            sr.print_summary(vm, canonical_model, vp.name)

    # Aggregate video metrics across all frames of all videos
    if not all_frames:
        agg = None
    else:
        temporal = [m["temporal_iou"] for m in all_frames if "temporal_iou" in m]
        num_det = [m.get("num_detections", 0) for m in all_frames]
        inf_ms = [m.get("inference_time_ms", 0.0) for m in all_frames]
        fps = [m.get("fps", 0.0) for m in all_frames]

        no_det_rate = float(np.mean([1.0 if d == 0 else 0.0 for d in num_det])) if num_det else 0.0
        mean_temporal = float(np.mean(temporal)) if temporal else 0.0
        mean_inf_ms = float(np.mean(inf_ms)) if inf_ms else 0.0
        mean_fps = float(np.mean(fps)) if fps else 0.0

        agg = {
            "n_videos": len(vids),
            "n_frames_total": len(all_frames),
            "mean_temporal_iou": mean_temporal,
            "no_detection_rate": no_det_rate,
            "frame_to_frame_stability": mean_temporal,  # same signal as temporal IoU
            "mean_inference_time_ms": mean_inf_ms,
            "mean_fps": mean_fps,
        }

    if agg is None:
        passfail = None
        passed = None
    else:
        passfail = {
            "temporal_iou": agg["mean_temporal_iou"] >= VIDEO_THRESHOLDS["temporal_iou"],
            "no_detection_rate": agg["no_detection_rate"] <= VIDEO_THRESHOLDS["no_detection_rate"],
            "frame_to_frame_stability": agg["frame_to_frame_stability"] >= VIDEO_THRESHOLDS["frame_to_frame_stability"],
        }
        passed = bool(all(passfail.values()))

    return {
        "n_videos": len(vids),
        "videos": per_video,
        "video_aggregate": agg,
        "pass": passed,
        "pass_details": passfail,
    }


def run_image_benchmark(args, canonical_model: str) -> dict:
    _ensure_dirs()
    if not BENCH_IMAGES_RAW.is_dir():
        raise SystemExit("[ERROR] Missing `benchmark/images/raw/`. Run `python benchmark.py prepare` first.")

    images, _, label_dir = sr.collect_inputs(str(BENCH_IMAGES))
    if not images:
        raise SystemExit("[ERROR] No images found in `benchmark/images/raw/`. Run `python benchmark.py prepare` first.")
    if label_dir is None:
        raise SystemExit("[ERROR] No labels found in `benchmark/images/labeled/`.")

    seg = _build_segmentor(args, canonical_model)

    _maybe_warmup(seg, images, args.warmup)

    out_dir = Path(args.output) / canonical_model
    out_dir.mkdir(parents=True, exist_ok=True)

    img_metrics = []
    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

    for img_path in images:
        if args.save_outputs:
            m = sr.run_image(seg, img_path, out_dir, canonical_model, label_dir)
            if not m:
                continue
            # For aggregate FP/FN rates we need counts; compute once per image.
            lp = sr.find_label(img_path, label_dir)
            if lp and lp.exists():
                gt_mask, void_mask = sr.load_gt_mask(lp)
                if gt_mask is not None:
                    img = sr.cv2.imread(str(img_path))
                    result, _ = seg.infer(str(img_path))
                    h, w = img.shape[:2]
                    mask, _ = sr.extract_masks(result, h, w)
                    TP, TN, FP, FN = _compute_counts(mask, gt_mask, void_mask)
                    counts["TP"] += TP
                    counts["TN"] += TN
                    counts["FP"] += FP
                    counts["FN"] += FN
            img_metrics.append(m)
        else:
            m, c = _run_image_metrics_only(seg, img_path, label_dir)
            if not m:
                continue
            if c is not None:
                TP, TN, FP, FN = c
                counts["TP"] += TP
                counts["TN"] += TN
                counts["FP"] += FP
                counts["FN"] += FN
            img_metrics.append(m)

    if not img_metrics:
        raise SystemExit("[ERROR] No image metrics collected.")

    sr.print_summary(img_metrics, canonical_model, f"{len(img_metrics)} image(s) [benchmark]")

    agg = {
        "n_images": len(img_metrics),
        "mean_iou": float(np.mean([m["iou"] for m in img_metrics if "iou" in m])),
        "mean_f1": float(np.mean([m["f1"] for m in img_metrics if "f1" in m])),
        "mean_pixel_accuracy": float(np.mean([m["pixel_accuracy"] for m in img_metrics if "pixel_accuracy" in m])),
        "counts": counts,
    }
    agg.update(_false_rates_from_counts(counts["TP"], counts["TN"], counts["FP"], counts["FN"]))

    passfail = {
        "iou": agg["mean_iou"] >= IMAGE_THRESHOLDS["iou"],
        "f1": agg["mean_f1"] >= IMAGE_THRESHOLDS["f1"],
        "pixel_accuracy": agg["mean_pixel_accuracy"] >= IMAGE_THRESHOLDS["pixel_accuracy"],
        "false_negative_rate": agg["false_negative_rate"] <= IMAGE_THRESHOLDS["false_negative_rate"],
        "false_positive_rate": agg["false_positive_rate"] <= IMAGE_THRESHOLDS["false_positive_rate"],
    }

    report = {
        "model": canonical_model,
        "images": img_metrics,
        "image_aggregate": agg,
        "thresholds": {"images": IMAGE_THRESHOLDS, "videos": VIDEO_THRESHOLDS},
        "pass": bool(all(passfail.values())),
        "pass_details": passfail,
    }

    rp = out_dir / f"benchmark_metrics_{canonical_model}.json"
    rp.write_text(json.dumps(report, indent=2))
    print(f"\nBenchmark JSON report → {rp.resolve()}")

    return report


def write_benchmark_csv(*, out_csv: Path, model: str, image_report: dict, video_report: dict) -> None:
    """
    Write a single CSV summary with threshold comparisons, matching the benchmark
    tables in the provided slides (images + videos + Jetson perf where measurable).
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    img_agg = image_report.get("image_aggregate") or {}
    vid_agg = video_report.get("video_aggregate") or {}

    rows = []

    # Images (200 images)
    rows.append(
        {
            "section": "images",
            "metric": "mIoU / IoU",
            "value": img_agg.get("mean_iou", ""),
            "threshold": IMAGE_THRESHOLDS["iou"],
            "pass": image_report.get("pass_details", {}).get("iou", ""),
        }
    )
    rows.append(
        {
            "section": "images",
            "metric": "F1 Score",
            "value": img_agg.get("mean_f1", ""),
            "threshold": IMAGE_THRESHOLDS["f1"],
            "pass": image_report.get("pass_details", {}).get("f1", ""),
        }
    )
    rows.append(
        {
            "section": "images",
            "metric": "Pixel Accuracy",
            "value": img_agg.get("mean_pixel_accuracy", ""),
            "threshold": IMAGE_THRESHOLDS["pixel_accuracy"],
            "pass": image_report.get("pass_details", {}).get("pixel_accuracy", ""),
        }
    )
    rows.append(
        {
            "section": "images",
            "metric": "False Negative Rate",
            "value": img_agg.get("false_negative_rate", ""),
            "threshold": IMAGE_THRESHOLDS["false_negative_rate"],
            "pass": image_report.get("pass_details", {}).get("false_negative_rate", ""),
        }
    )
    rows.append(
        {
            "section": "images",
            "metric": "False Positive Rate",
            "value": img_agg.get("false_positive_rate", ""),
            "threshold": IMAGE_THRESHOLDS["false_positive_rate"],
            "pass": image_report.get("pass_details", {}).get("false_positive_rate", ""),
        }
    )

    # Videos (10 videos)
    rows.append(
        {
            "section": "videos",
            "metric": "Temporal IoU",
            "value": vid_agg.get("mean_temporal_iou", ""),
            "threshold": VIDEO_THRESHOLDS["temporal_iou"],
            "pass": video_report.get("pass_details", {}).get("temporal_iou", ""),
        }
    )
    rows.append(
        {
            "section": "videos",
            "metric": "No-detection Rate",
            "value": vid_agg.get("no_detection_rate", ""),
            "threshold": VIDEO_THRESHOLDS["no_detection_rate"],
            "pass": video_report.get("pass_details", {}).get("no_detection_rate", ""),
        }
    )
    rows.append(
        {
            "section": "videos",
            "metric": "Frame-to-frame Stability",
            "value": vid_agg.get("frame_to_frame_stability", ""),
            "threshold": VIDEO_THRESHOLDS["frame_to_frame_stability"],
            "pass": video_report.get("pass_details", {}).get("frame_to_frame_stability", ""),
        }
    )

    # Jetson performance table (fill what we can from measured timings)
    # Note: GPU/CPU/RAM utilization require tegrastats; left blank here.
    rows.append(
        {
            "section": "jetson",
            "metric": "GPU Utilization (%)",
            "value": "",
            "threshold": JETSON_THRESHOLDS["gpu_utilization_pct"],
            "pass": "",
        }
    )
    rows.append(
        {
            "section": "jetson",
            "metric": "CPU Utilization (%)",
            "value": "",
            "threshold": JETSON_THRESHOLDS["cpu_utilization_pct"],
            "pass": "",
        }
    )
    rows.append(
        {
            "section": "jetson",
            "metric": "RAM Usage (%)",
            "value": "",
            "threshold": JETSON_THRESHOLDS["ram_usage_pct"],
            "pass": "",
        }
    )
    rows.append(
        {
            "section": "jetson",
            "metric": "Image Inference Speed (ms)",
            "value": img_agg.get("mean_inference_time_ms", ""),
            "threshold": JETSON_THRESHOLDS["image_inference_speed_ms"],
            "pass": "",
        }
    )
    rows.append(
        {
            "section": "jetson",
            "metric": "Video Inference Speed (ms)",
            "value": vid_agg.get("mean_inference_time_ms", ""),
            "threshold": JETSON_THRESHOLDS["video_inference_speed_ms"],
            "pass": "",
        }
    )
    rows.append(
        {
            "section": "jetson",
            "metric": "Video FPS",
            "value": vid_agg.get("mean_fps", ""),
            "threshold": JETSON_THRESHOLDS["video_fps"],
            "pass": "",
        }
    )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "section", "metric", "value", "threshold", "pass"])
        w.writeheader()
        for r in rows:
            r2 = dict(r)
            r2["model"] = model
            w.writerow(r2)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare and run an off-road segmentation benchmark.")
    sub = p.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare", help="Create benchmark/images from test_data.")
    prep.add_argument("--n-images", type=int, default=200, help="Number of labeled images to sample/copy.")
    prep.add_argument("--seed", type=int, default=1337, help="Random seed for sampling.")
    prep.add_argument(
        "--source",
        default="orfd",
        choices=["orfd", "test_data"],
        help="Where to sample labeled image pairs from.",
    )
    prep.add_argument(
        "--source-root",
        default=str(Path("datasets") / "ORFD"),
        help="Root folder for the selected --source (default: datasets/ORFD).",
    )
    prep.add_argument("--clear", action="store_true", help="Clear existing benchmark/images contents first.")

    run = sub.add_parser("run", help="Run benchmark on prepared images.")
    run.add_argument(
        "--model",
        required=True,
        choices=sorted(set(MODEL_ALIASES.keys())),
        help="Model backend (aliases supported): sam3, sam2, yoloe26, yolo26, sam21, etc.",
    )
    run.add_argument("--output", default="output", help="Base output directory for annotated results/reports.")
    run.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (YOLO/SAM3).")
    run.add_argument("--prompts", nargs="+", default=None, help="Custom text prompts (YOLO/SAM3 only).")
    run.add_argument("--model-size", default="x", choices=["n", "s", "m", "l", "x"], help="YOLOE-26 size variant.")
    run.add_argument(
        "--yolo-weights",
        default=None,
        help="Path to a local YOLOE-26 .pt weights file (disables any auto-download behavior).",
    )
    run.add_argument("--sam3-weights", default="sam3.pt", help="Path to sam3.pt.")
    run.add_argument("--sam21-weights", default="sam2.1_l.pt", help="Path to sam2.1_l.pt.")
    run.add_argument(
        "--hardware",
        default="auto",
        choices=["auto", "jetson", "desktop"],
        help="Hardware preset. 'jetson' reduces disk I/O by default.",
    )
    run.add_argument(
        "--save-outputs",
        action="store_true",
        help="Save annotated images to output/<model>/ (slower; more disk I/O).",
    )
    run.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup images before timing (best-effort).",
    )

    allcmd = sub.add_parser("all", help="Run images + videos and write CSV summary.")
    allcmd.add_argument(
        "--model",
        required=True,
        choices=sorted(set(MODEL_ALIASES.keys())),
        help="Model backend (aliases supported): sam3, sam2, yoloe26, yolo26, sam21, etc.",
    )
    allcmd.add_argument("--output", default="output", help="Base output directory for outputs/reports.")
    allcmd.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (YOLO/SAM3).")
    allcmd.add_argument("--prompts", nargs="+", default=None, help="Custom text prompts (YOLO/SAM3 only).")
    allcmd.add_argument("--model-size", default="x", choices=["n", "s", "m", "l", "x"], help="YOLOE-26 size variant.")
    allcmd.add_argument(
        "--yolo-weights",
        default=None,
        help="Path to a local YOLOE-26 .pt weights file (disables any auto-download behavior).",
    )
    allcmd.add_argument("--sam3-weights", default="sam3.pt", help="Path to sam3.pt.")
    allcmd.add_argument("--sam21-weights", default="sam2.1_l.pt", help="Path to sam2.1_l.pt.")
    allcmd.add_argument(
        "--hardware",
        default="auto",
        choices=["auto", "jetson", "desktop"],
        help="Hardware preset. 'jetson' reduces disk I/O by default.",
    )
    allcmd.add_argument(
        "--save-outputs",
        action="store_true",
        help="Save annotated images and videos (slower; more disk I/O).",
    )
    allcmd.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup images before timing (best-effort).",
    )

    return p


def main() -> None:
    args = build_parser().parse_args()
    _ensure_dirs()

    if args.cmd == "prepare":
        meta = prepare_benchmark_images(
            source_root=Path(args.source_root),
            n_images=args.n_images,
            seed=args.seed,
            clear=args.clear,
            source=args.source,
        )
        print(json.dumps(meta, indent=2))
        return

    canonical = _canonical_model(getattr(args, "model", ""))
    # Hardware presets (Jetson Orin NX / NRU-51V+): minimize disk I/O by default.
    if args.cmd in ("run", "all"):
        is_jetson = detect_jetson()
        if args.hardware == "jetson" or (args.hardware == "auto" and is_jetson):
            # On embedded devices, default to metrics-only unless user explicitly opts in.
            if not args.save_outputs:
                args.save_outputs = False
        else:
            # On desktop, keeping old behavior (save annotated outputs) is convenient.
            if not args.save_outputs:
                args.save_outputs = True

    if args.cmd == "run":
        run_image_benchmark(args, canonical)
        return

    if args.cmd == "all":
        # Build segmentor once; used for both images and videos.
        seg = _build_segmentor(args, canonical)

        print("\n[Benchmark: Images]")
        image_report = run_image_benchmark(args, canonical)

        print("\n[Benchmark: Videos]")
        video_report = run_video_benchmark(args, canonical, seg)

        out_dir = Path(args.output) / canonical
        out_dir.mkdir(parents=True, exist_ok=True)

        # Write combined JSON + CSV
        combined = {
            "model": canonical,
            "thresholds": {"images": IMAGE_THRESHOLDS, "videos": VIDEO_THRESHOLDS, "jetson": JETSON_THRESHOLDS},
            "images": image_report,
            "videos": video_report,
        }
        (out_dir / f"benchmark_all_{canonical}.json").write_text(json.dumps(combined, indent=2))

        csv_path = out_dir / f"benchmark_all_{canonical}.csv"
        write_benchmark_csv(out_csv=csv_path, model=canonical, image_report=image_report, video_report=video_report)
        print(f"\nCSV summary → {csv_path.resolve()}")
        return

    raise SystemExit(f"[ERROR] Unsupported command: {args.cmd!r}")


if __name__ == "__main__":
    main()

