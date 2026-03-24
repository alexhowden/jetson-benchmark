#!/usr/bin/env bash
set -euo pipefail

# Run end-to-end benchmark.py all for all supported models.
# Keeps full pipeline behavior (inference + overlay + video writing when enabled).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_ROOT="output"
HARDWARE="jetson"
VIDEO_WRITER="auto"
VIDEO_CAPTURE="auto"
YOLO_MODE="compare"
YOLO_ENGINE=""
WARMUP=3
SAVE_OUTPUTS=0
SKIP_PREPARE=0
N_IMAGES=200
SEED=1337
SOURCE="orfd"
SOURCE_ROOT="datasets/ORFD"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_all_benchmarks.sh [options]

Options:
  --output-root <dir>       Base output directory (default: output)
  --hardware <auto|jetson|desktop>
                            Hardware preset for benchmark.py (default: jetson)
  --video-writer <auto|opencv|gstreamer>
                            Video writer backend when saving videos (default: auto)
  --video-capture <auto|opencv|gstreamer>
                            Video capture backend when reading videos (default: auto)
  --yolo-mode <compare|pytorch|tensorrt>
                            YOLO run mode: compare runs pytorch+tensorrt side by side
                            (default: compare)
  --yolo-engine <path>      TensorRT engine path (optional, for yolo26 runs)
  --warmup <n>              Warmup images per run (default: 3)
  --save-outputs            Enable annotated image/video saving
  --no-save-outputs         Disable annotated image/video saving (default)
  --skip-prepare            Skip benchmark.py prepare step
  --n-images <n>            Number of benchmark images to prepare (default: 200)
  --seed <n>                Random seed for prepare sampling (default: 1337)
  --source <orfd|test_data> Prepare source (default: orfd)
  --source-root <path>      Source root for prepare (default: datasets/ORFD)
  -h, --help                Show this help

Notes:
- This script runs model variants:
  yolo26(x/pytorch), yolo26(x/tensorrt), yolo26(n/pytorch), yolo26(n/tensorrt),
  sam21, sam3, fastsam, mobilesam
- In --yolo-mode compare, YOLO outputs are separated by backend to avoid overwriting.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)
      OUTPUT_ROOT="$2"; shift 2 ;;
    --hardware)
      HARDWARE="$2"; shift 2 ;;
    --video-writer)
      VIDEO_WRITER="$2"; shift 2 ;;
    --video-capture)
      VIDEO_CAPTURE="$2"; shift 2 ;;
    --yolo-mode)
      YOLO_MODE="$2"; shift 2 ;;
    --yolo-engine)
      YOLO_ENGINE="$2"; shift 2 ;;
    --warmup)
      WARMUP="$2"; shift 2 ;;
    --save-outputs)
      SAVE_OUTPUTS=1; shift ;;
    --no-save-outputs)
      SAVE_OUTPUTS=0; shift ;;
    --skip-prepare)
      SKIP_PREPARE=1; shift ;;
    --n-images)
      N_IMAGES="$2"; shift 2 ;;
    --seed)
      SEED="$2"; shift 2 ;;
    --source)
      SOURCE="$2"; shift 2 ;;
    --source-root)
      SOURCE_ROOT="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      usage
      exit 2 ;;
  esac
done

case "$YOLO_MODE" in
  compare|pytorch|tensorrt) ;;
  *)
    echo "[ERROR] Invalid --yolo-mode: $YOLO_MODE (expected: compare|pytorch|tensorrt)" >&2
    exit 2
    ;;
esac

if [[ $SKIP_PREPARE -eq 0 ]]; then
  if [[ ! -d "$SOURCE_ROOT" && -d "benchmark/images/raw" && -d "benchmark/images/labeled" ]]; then
    echo "[STEP] Prepare source '$SOURCE_ROOT' not found; using existing benchmark/images and skipping prepare."
  else
    echo "[STEP] Preparing benchmark image set..."
    python benchmark.py prepare \
      --n-images "$N_IMAGES" \
      --seed "$SEED" \
      --source "$SOURCE" \
      --source-root "$SOURCE_ROOT"
  fi
fi

SAVE_FLAG=()
if [[ $SAVE_OUTPUTS -eq 1 ]]; then
  SAVE_FLAG=(--save-outputs)
fi

run_model() {
  local model="$1"
  local out_root="$2"
  shift 2

  echo
  echo "[RUN] model=$model output=$out_root"

  python benchmark.py all \
    --model "$model" \
    --output "$out_root" \
    --hardware "$HARDWARE" \
    --warmup "$WARMUP" \
    --video-writer "$VIDEO_WRITER" \
    --video-capture "$VIDEO_CAPTURE" \
    "${SAVE_FLAG[@]}" \
    "$@"
}

# YOLO26 x and n are separated by size + backend to keep reports side by side.
run_yolo_backend() {
  local backend="$1"
  local size="$2"
  local out_root="$OUTPUT_ROOT/yolo26-$size-$backend"
  local extra=(--model-size "$size" --yolo-backend "$backend")

  if [[ "$backend" == "tensorrt" && -n "$YOLO_ENGINE" ]]; then
    extra+=(--yolo-engine "$YOLO_ENGINE")
  fi

  run_model yolo26 "$out_root" "${extra[@]}"
}

if [[ "$YOLO_MODE" == "compare" || "$YOLO_MODE" == "pytorch" ]]; then
  run_yolo_backend pytorch x
  run_yolo_backend pytorch n
fi

if [[ "$YOLO_MODE" == "compare" || "$YOLO_MODE" == "tensorrt" ]]; then
  run_yolo_backend tensorrt x
  run_yolo_backend tensorrt n
fi

run_model sam21 "$OUTPUT_ROOT/sam21"
run_model sam3 "$OUTPUT_ROOT/sam3"
run_model fastsam "$OUTPUT_ROOT/fastsam"
run_model mobilesam "$OUTPUT_ROOT/mobilesam"

echo
echo "[DONE] All benchmark runs finished."
