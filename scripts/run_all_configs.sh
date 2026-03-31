#!/bin/bash
#
# Comprehensive YOLOE-26 Configuration Benchmark Runner
#
# Tests all combinations of:
#   - Model size: n (nano), x (extra-large)
#   - Backend: pytorch, tensorrt
#
# Image size is fixed at 640 (default)
# Video backend is fixed at opencv (GStreamer not available)
#
# Usage:
#   ./run_all_configs.sh
#   ./run_all_configs.sh --skip-n          # Skip YOLO-26n tests
#   ./run_all_configs.sh --skip-x          # Skip YOLO-26x tests
#   ./run_all_configs.sh --only-tensorrt   # Only test TensorRT backend
#   ./run_all_configs.sh --only-pytorch    # Only test PyTorch backend
#   ./run_all_configs.sh --rebuild-engines # Force TensorRT engine re-export
#   ./run_all_configs.sh --trt-half        # Use FP16 for TensorRT export (with --rebuild-engines)
#   ./run_all_configs.sh --save-outputs    # Save annotated image/video outputs
#

set -e  # Exit on error

# Prefer python3 in containers where `python` may not be installed.
if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "[ERROR] Neither python3 nor python was found in PATH"
    exit 1
fi

# Parse arguments
SKIP_N=false
SKIP_X=false
ONLY_TENSORRT=false
ONLY_PYTORCH=false
REBUILD_ENGINES=false
TRT_HALF=false
SAVE_OUTPUTS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-n)
            SKIP_N=true
            shift
            ;;
        --skip-x)
            SKIP_X=true
            shift
            ;;
        --only-tensorrt)
            ONLY_TENSORRT=true
            shift
            ;;
        --only-pytorch)
            ONLY_PYTORCH=true
            shift
            ;;
        --rebuild-engines)
            REBUILD_ENGINES=true
            shift
            ;;
        --trt-half)
            TRT_HALF=true
            shift
            ;;
        --save-outputs)
            SAVE_OUTPUTS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--skip-n] [--skip-x] [--only-tensorrt] [--only-pytorch] [--rebuild-engines] [--trt-half] [--save-outputs]"
            exit 1
            ;;
    esac
done

# Configuration
BASE_OUTPUT="output/config-tests"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${BASE_OUTPUT}/logs"
mkdir -p "${LOG_DIR}"

# Summary file
SUMMARY_FILE="${BASE_OUTPUT}/summary_${TIMESTAMP}.txt"
COMPARISON_CSV="${BASE_OUTPUT}/comparison_${TIMESTAMP}.csv"

echo "================================================================================"
echo "YOLOE-26 Configuration Benchmark Suite"
echo "================================================================================"
echo ""
echo "Output directory: ${BASE_OUTPUT}"
echo "Summary file: ${SUMMARY_FILE}"
echo "Comparison CSV: ${COMPARISON_CSV}"
echo ""

# Initialize summary
cat > "${SUMMARY_FILE}" << EOF
YOLOE-26 Configuration Benchmark Results
Generated: $(date)
================================================================================

EOF

# Initialize comparison CSV
cat > "${COMPARISON_CSV}" << EOF
config,model_size,backend,status,elapsed_sec,img_mean_iou,img_mean_f1,img_pixel_acc,img_fnr,img_fpr,img_inference_ms,img_fps,vid_temporal_iou,vid_no_detect_rate,vid_stability,vid_inference_ms,vid_fps,img_pass,vid_pass
EOF

# Function to run a single benchmark
run_benchmark() {
    local MODEL_SIZE=$1
    local BACKEND=$2

    # Build config name (imgsz=640, video_backend=opencv)
    local CONFIG_NAME="yolo26-${MODEL_SIZE}-${BACKEND}"
    local OUTPUT_DIR="${BASE_OUTPUT}/${CONFIG_NAME}"
    local LOG_FILE="${LOG_DIR}/${CONFIG_NAME}.log"

    echo "--------------------------------------------------------------------------------"
    echo "Running: ${CONFIG_NAME}"
    echo "Log: ${LOG_FILE}"
    echo "--------------------------------------------------------------------------------"

    # Build command (imgsz=640 default, video-capture=opencv)
    local CMD="${PYTHON_BIN} benchmark.py all --model yolo26 --model-size ${MODEL_SIZE} --yolo-backend ${BACKEND} --output ${OUTPUT_DIR} --video-capture opencv"
    if [ "${BACKEND}" = "tensorrt" ] && [ "${REBUILD_ENGINES}" = true ]; then
        # Keep TensorRT comparisons fair by rebuilding engine from current prompts/settings.
        CMD+=" --yolo-rebuild-engine"
        if [ "${TRT_HALF}" = true ]; then
            CMD+=" --yolo-trt-half"
        fi
    fi
    if [ "${SAVE_OUTPUTS}" = true ]; then
        CMD+=" --save-outputs"
    fi

    echo "Command: ${CMD}"
    echo ""

    # Run benchmark
    local START_TIME=$(date +%s)
    local STATUS="SUCCESS"

    if ${CMD} > "${LOG_FILE}" 2>&1; then
        echo "✅ ${CONFIG_NAME} completed successfully"
    else
        echo "❌ ${CONFIG_NAME} FAILED (see ${LOG_FILE})"
        STATUS="FAILED"
    fi

    local END_TIME=$(date +%s)
    local ELAPSED=$((END_TIME - START_TIME))

    # Extract metrics from JSON
    local JSON_FILE="${OUTPUT_DIR}/yolo26/benchmark_all_yolo26.json"
    local CSV_FILE="${OUTPUT_DIR}/yolo26/benchmark_all_yolo26.csv"

    if [ -f "${JSON_FILE}" ] && [ "${STATUS}" = "SUCCESS" ]; then
        # Extract metrics using Python
        local METRICS=$(python3 << EOF
import json
import sys

try:
    with open("${JSON_FILE}") as f:
        data = json.load(f)

    img = data.get("images", {})
    vid = data.get("videos", {})
    img_agg = img.get("image_aggregate", {})
    vid_agg = vid.get("video_aggregate", {})

    print(f"{img_agg.get('mean_iou', 0):.4f}", end=",")
    print(f"{img_agg.get('mean_f1', 0):.4f}", end=",")
    print(f"{img_agg.get('mean_pixel_accuracy', 0):.4f}", end=",")
    print(f"{img_agg.get('false_negative_rate', 0):.4f}", end=",")
    print(f"{img_agg.get('false_positive_rate', 0):.4f}", end=",")

    # Calculate image FPS from inference time
    img_inf_ms = img_agg.get('mean_inference_time_ms', 0)
    img_fps = 1000.0 / img_inf_ms if img_inf_ms > 0 else 0
    print(f"{img_inf_ms:.2f}", end=",")
    print(f"{img_fps:.2f}", end=",")

    print(f"{vid_agg.get('mean_temporal_iou', 0):.4f}", end=",")
    print(f"{vid_agg.get('no_detection_rate', 0):.4f}", end=",")
    print(f"{vid_agg.get('frame_to_frame_stability', 0):.4f}", end=",")
    print(f"{vid_agg.get('mean_inference_time_ms', 0):.2f}", end=",")
    print(f"{vid_agg.get('mean_fps', 0):.2f}", end=",")

    print(f"{img.get('pass', False)}", end=",")
    print(f"{vid.get('pass', False)}")

except Exception as e:
    print("0,0,0,0,0,0,0,0,0,0,0,0,False,False", file=sys.stderr)
    sys.exit(1)
EOF
)

        # Append to comparison CSV
        echo "${CONFIG_NAME},${MODEL_SIZE},${BACKEND},${STATUS},${ELAPSED},${METRICS}" >> "${COMPARISON_CSV}"

        # Copy full CSV to summary location
        if [ -f "${CSV_FILE}" ]; then
            cp "${CSV_FILE}" "${BASE_OUTPUT}/${CONFIG_NAME}_full.csv"
        fi
    else
        # Failed run - add placeholder
        echo "${CONFIG_NAME},${MODEL_SIZE},${BACKEND},${STATUS},${ELAPSED},0,0,0,0,0,0,0,0,0,0,0,0,False,False" >> "${COMPARISON_CSV}"
    fi

    # Append to summary
    cat >> "${SUMMARY_FILE}" << EOF
${CONFIG_NAME}
  Status: ${STATUS}
  Elapsed: ${ELAPSED}s
  Output: ${OUTPUT_DIR}
  Log: ${LOG_FILE}

EOF

    echo ""
}

# Run all configurations
TOTAL=0
COMPLETED=0

# YOLO-26n configurations
if [ "$SKIP_N" = false ]; then
    # TensorRT backend
    if [ "$ONLY_PYTORCH" = false ]; then
        run_benchmark "n" "tensorrt"
        TOTAL=$((TOTAL + 1))
    fi

    # PyTorch backend
    if [ "$ONLY_TENSORRT" = false ]; then
        run_benchmark "n" "pytorch"
        TOTAL=$((TOTAL + 1))
    fi
fi

# YOLO-26x configurations
if [ "$SKIP_X" = false ]; then
    # TensorRT backend
    if [ "$ONLY_PYTORCH" = false ]; then
        run_benchmark "x" "tensorrt"
        TOTAL=$((TOTAL + 1))
    fi

    # PyTorch backend
    if [ "$ONLY_TENSORRT" = false ]; then
        run_benchmark "x" "pytorch"
        TOTAL=$((TOTAL + 1))
    fi
fi

# Final summary
echo "================================================================================"
echo "BENCHMARK SUITE COMPLETE"
echo "================================================================================"
echo ""
echo "Results saved to:"
echo "  Summary: ${SUMMARY_FILE}"
echo "  Comparison CSV: ${COMPARISON_CSV}"
echo "  Full CSVs: ${BASE_OUTPUT}/*_full.csv"
echo "  Logs: ${LOG_DIR}/"
echo ""
echo "View comparison:"
echo "  cat ${COMPARISON_CSV} | column -t -s,"
echo ""
