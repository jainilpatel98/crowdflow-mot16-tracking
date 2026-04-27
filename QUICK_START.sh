#!/bin/bash
# =============================================================================
# QUICK_START.sh — MOT16 YOLO-to-ResNet50 Distillation Pipeline
# =============================================================================
#
# USAGE:  ./QUICK_START.sh [command] [--gpus N]
#
# OPTIONS:
#   --gpus N    Number of GPUs to use for torchrun (default: 4).
#               Overrides the NGPU environment variable.
#               Example:  ./QUICK_START.sh train --gpus 2
#
# COMMANDS:
#   train       Full 80-epoch ResNet50 distillation (default, recommended)
#   smoke       2-epoch smoke test to verify the pipeline works
#   cache       (optional) Pre-compute teacher cache, then train with cache
#   eval        Evaluate detection quality on MOT16 val sequences
#   track       Run tracking on a single MOT16 sequence
#   monitor     Watch live training metrics
#   export      Export best checkpoint to ONNX
#   check       Check if a training process is currently running
#   help        Show this help
#
# =============================================================================
# CACHE vs LIVE TEACHER — how to control it:
#
#   Option A (recommended): YAML config key  ← persistent per-config setting
#     In any configs/*.yaml, set:
#       teacher:
#         use_cache: false    # live teacher (default)
#         use_cache: true     # use pre-computed cache
#
#   Option B: CLI flag  ← one-off override, does NOT change the config file
#     python tools/train_student.py --config configs/student_distill_resnet50.yaml --use-cache
#
#   The CLI flag --use-cache is OR-ed with the YAML value, so either one
#   being true enables cache mode. Default for both is false (live teacher).
# =============================================================================

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv/bin/python"
# If no .venv, fall back to whatever python3 is active
PYTHON="${VENV:-python3}"
NGPU="${NGPU:-4}"    # default 4; override via --gpus N flag or NGPU=N env var

# ── helpers ──────────────────────────────────────────────────────────────────

_header() { echo ""; echo "══════════════════════════════════════════════════════"; echo "  $1"; echo "══════════════════════════════════════════════════════"; echo ""; }
_cmd()    { echo "  ▶  $*"; }

check_training() {
    _header "Active Training Processes"
    local count
    count=$(ps aux | grep -E "torchrun|train_student" | grep -v grep | wc -l | tr -d ' ')
    if [ "$count" -gt 0 ]; then
        echo "  ✅  $count training process(es) running:"
        ps aux | grep -E "torchrun|train_student" | grep -v grep
    else
        echo "  ⬜  No training processes found."
    fi
    echo ""
}

# ── commands ─────────────────────────────────────────────────────────────────

do_train() {
    _header "Full ResNet50 Distillation Training (80 epochs, live teacher)"
    # ─── COMMAND ───────────────────────────────────────────────────────────
    # Default: live teacher — teacher and student share the same augmented input
    #   torchrun --nproc_per_node=4 tools/train_student.py \
    #       --config configs/student_distill_resnet50.yaml
    #
    # To resume from a checkpoint:
    #   torchrun --nproc_per_node=4 tools/train_student.py \
    #       --config configs/student_distill_resnet50.yaml \
    #       --resume runs/student_distill_resnet50/last.pt
    #
    # To enable cache mode (faster, slight accuracy trade-off):
    #   torchrun --nproc_per_node=4 tools/train_student.py \
    #       --config configs/student_distill_resnet50.yaml \
    #       --use-cache
    # ───────────────────────────────────────────────────────────────────────
    cd "$ROOT"
    _cmd "torchrun --nproc_per_node=$NGPU tools/train_student.py --config configs/student_distill_resnet50.yaml"
    torchrun --nproc_per_node="$NGPU" tools/train_student.py \
        --config configs/student_distill_resnet50.yaml
}

do_smoke() {
    _header "Smoke Test (2 epochs — verifies pipeline without full training)"
    # ─── COMMAND ───────────────────────────────────────────────────────────
    #   python tools/train_student.py \
    #       --config configs/student_distill_resnet50_smoke.yaml
    # ───────────────────────────────────────────────────────────────────────
    cd "$ROOT"
    _cmd "python tools/train_student.py --config configs/student_distill_resnet50_smoke.yaml"
    "$PYTHON" tools/train_student.py \
        --config configs/student_distill_resnet50_smoke.yaml
}

do_cache() {
    _header "Pre-compute Teacher Cache  (optional, for cache-mode training)"
    echo "  NOTE: Default training is live-teacher (no cache needed)."
    echo "  Use cache only if GPU memory is constrained or teacher is slow."
    echo ""
    # ─── COMMANDS ──────────────────────────────────────────────────────────
    # Step 1: cache train split
    #   python tools/cache_teacher_outputs.py \
    #       --config configs/student_distill_resnet50.yaml --split train
    #
    # Step 2: cache val split
    #   python tools/cache_teacher_outputs.py \
    #       --config configs/student_distill_resnet50.yaml --split val
    #
    # Step 3: train with cache (--use-cache flag)
    #   torchrun --nproc_per_node=4 tools/train_student.py \
    #       --config configs/student_distill_resnet50.yaml --use-cache
    # ───────────────────────────────────────────────────────────────────────
    cd "$ROOT"
    _cmd "python tools/cache_teacher_outputs.py --config configs/student_distill_resnet50.yaml --split train"
    "$PYTHON" tools/cache_teacher_outputs.py \
        --config configs/student_distill_resnet50.yaml --split train

    _cmd "python tools/cache_teacher_outputs.py --config configs/student_distill_resnet50.yaml --split val"
    "$PYTHON" tools/cache_teacher_outputs.py \
        --config configs/student_distill_resnet50.yaml --split val

    echo ""
    echo "  Cache written to: data/teacher_cache/"
    echo "  Now run:  ./QUICK_START.sh train-cache   (or pass --use-cache manually)"
}

do_train_cache() {
    _header "Train with Pre-computed Cache"
    # ─── COMMAND ───────────────────────────────────────────────────────────
    #   torchrun --nproc_per_node=4 tools/train_student.py \
    #       --config configs/student_distill_resnet50.yaml \
    #       --use-cache
    # ───────────────────────────────────────────────────────────────────────
    cd "$ROOT"
    _cmd "torchrun --nproc_per_node=$NGPU tools/train_student.py --config configs/student_distill_resnet50.yaml --use-cache"
    torchrun --nproc_per_node="$NGPU" tools/train_student.py \
        --config configs/student_distill_resnet50.yaml \
        --use-cache
}

do_eval() {
    local model_target="${1:-}"
    cd "$ROOT"
    if [ "$model_target" = "teacher" ]; then
        local teacher_ckpt="${2:-}"
        _header "Evaluate Teacher Detection on MOT16 Val Sequences"
        if [ -n "$teacher_ckpt" ]; then
            _cmd "python tools/eval_detection.py --config configs/student_distill_resnet50.yaml --model-type teacher --checkpoint $teacher_ckpt"
            "$PYTHON" tools/eval_detection.py \
                --config configs/student_distill_resnet50.yaml \
                --model-type teacher \
                --checkpoint "$teacher_ckpt"
        else
            _cmd "python tools/eval_detection.py --config configs/student_distill_resnet50.yaml --model-type teacher"
            "$PYTHON" tools/eval_detection.py \
                --config configs/student_distill_resnet50.yaml \
                --model-type teacher
        fi
    else
        # ─── COMMAND ───────────────────────────────────────────────────────
        #   python tools/eval_detection.py \
        #       --config configs/student_distill_resnet50.yaml \
        #       --checkpoint runs/student_distill_resnet50/best.pt
        # ───────────────────────────────────────────────────────────────────
        local ckpt="${1:-runs/student_distill_resnet50/best.pt}"
        _header "Evaluate Detection on MOT16 Val Sequences"
        _cmd "python tools/eval_detection.py --config configs/student_distill_resnet50.yaml --checkpoint $ckpt"
        "$PYTHON" tools/eval_detection.py \
            --config configs/student_distill_resnet50.yaml \
            --checkpoint "$ckpt"
    fi
}

do_track() {
    _header "Run Tracking on a MOT16 Sequence"
    # ─── COMMAND ───────────────────────────────────────────────────────────
    #   python tools/eval_tracking.py \
    #       --config   configs/student_distill_resnet50.yaml \
    #       --tracker-config configs/tracker.yaml \
    #       --checkpoint runs/student_distill_resnet50/best.pt \
    #       --sequence-dir MOT16/train/MOT16-10 \
    #       --output outputs/student_tracking/MOT16-10.txt \
    #       --output-video outputs/student_tracking/MOT16-10.mp4
    #
    # Available sequences (val split): MOT16-05, MOT16-10
    # ───────────────────────────────────────────────────────────────────────
    local seq="${1:-MOT16/train/MOT16-10}"
    local out="outputs/student_tracking/$(basename "$seq").txt"
    local out_video="outputs/student_tracking/$(basename "$seq").mp4"
    local ckpt="${2:-runs/student_distill_resnet50/best.pt}"
    cd "$ROOT"
    mkdir -p "$(dirname "$out")"
    _cmd "python tools/eval_tracking.py --sequence-dir $seq --output $out --output-video $out_video"
    "$PYTHON" tools/eval_tracking.py \
        --config configs/student_distill_resnet50.yaml \
        --tracker-config configs/tracker.yaml \
        --checkpoint "$ckpt" \
        --sequence-dir "$seq" \
        --output "$out" \
        --output-video "$out_video"
    echo ""
    echo "  Results saved to: $out"
    echo "  Tracking video saved to: $out_video"
}

do_monitor() {
    _header "Monitor Training (live metrics)"
    # ─── COMMAND ───────────────────────────────────────────────────────────
    #   python monitor_training.py runs/student_distill_resnet50
    # ───────────────────────────────────────────────────────────────────────
    cd "$ROOT"
    "$PYTHON" monitor_training.py runs/student_distill_resnet50
}

do_export() {
    _header "Export Best Checkpoint to ONNX"
    # ─── COMMAND ───────────────────────────────────────────────────────────
    #   python tools/export_onnx.py \
    #       --config configs/student_distill_resnet50.yaml \
    #       --checkpoint runs/student_distill_resnet50/best.pt \
    #       --output runs/student_distill_resnet50/student_resnet50.onnx
    # ───────────────────────────────────────────────────────────────────────
    cd "$ROOT"
    _cmd "python tools/export_onnx.py --checkpoint runs/student_distill_resnet50/best.pt"
    "$PYTHON" tools/export_onnx.py \
        --config configs/student_distill_resnet50.yaml \
        --checkpoint runs/student_distill_resnet50/best.pt \
        --output runs/student_distill_resnet50/student_resnet50.onnx
}

do_help() {
    _header "MOT16 Distillation Pipeline — Command Reference"
    cat <<'EOF'
  TRAINING
  ─────────────────────────────────────────────────────────────────────────────
  Full training (80 epochs, live teacher — default):
    ./QUICK_START.sh train
    # or directly:
    torchrun --nproc_per_node=4 tools/train_student.py \
        --config configs/student_distill_resnet50.yaml

  Resume from checkpoint:
    torchrun --nproc_per_node=4 tools/train_student.py \
        --config configs/student_distill_resnet50.yaml \
        --resume runs/student_distill_resnet50/last.pt

  Train with cache (set use_cache: true in YAML, OR pass --use-cache flag):
    ./QUICK_START.sh train-cache
    # or:
    torchrun --nproc_per_node=4 tools/train_student.py \
        --config configs/student_distill_resnet50.yaml --use-cache

  Smoke test (2 epochs, single process):
    ./QUICK_START.sh smoke
    # or:
    python tools/train_student.py \
        --config configs/student_distill_resnet50_smoke.yaml

  CACHE MODE (optional — only needed when use_cache: true)
  ─────────────────────────────────────────────────────────────────────────────
  Step 1 — cache train split:
    python tools/cache_teacher_outputs.py \
        --config configs/student_distill_resnet50.yaml --split train

  Step 2 — cache val split:
    python tools/cache_teacher_outputs.py \
        --config configs/student_distill_resnet50.yaml --split val

  EVALUATION
  ─────────────────────────────────────────────────────────────────────────────
  Detection metrics (mAP@0.5, precision, recall):
    ./QUICK_START.sh eval
    # or:
    python tools/eval_detection.py \
        --config configs/student_distill_resnet50.yaml \
        --checkpoint runs/student_distill_resnet50/best.pt

  Teacher detection metrics:
    ./QUICK_START.sh eval teacher
    # or:
    python tools/eval_detection.py \
        --config configs/student_distill_resnet50.yaml \
        --model-type teacher

  Tracking on a single sequence (MOT format output):
    ./QUICK_START.sh track MOT16/train/MOT16-10
    # or:
    python tools/eval_tracking.py \
        --config   configs/student_distill_resnet50.yaml \
        --tracker-config configs/tracker.yaml \
        --checkpoint runs/student_distill_resnet50/best.pt \
        --sequence-dir MOT16/train/MOT16-10 \
        --output outputs/student_tracking/MOT16-10.txt

  EXPORT
  ─────────────────────────────────────────────────────────────────────────────
  Export to ONNX:
    ./QUICK_START.sh export
    # or:
    python tools/export_onnx.py \
        --config configs/student_distill_resnet50.yaml \
        --checkpoint runs/student_distill_resnet50/best.pt \
        --output runs/student_distill_resnet50/student_resnet50.onnx

  UTILITIES
  ─────────────────────────────────────────────────────────────────────────────
  Monitor live training metrics:
    ./QUICK_START.sh monitor
    # or: python monitor_training.py runs/student_distill_resnet50

  Check running processes:
    ./QUICK_START.sh check

  ─────────────────────────────────────────────────────────────────────────────
  GPU COUNT (--gpus N):
    • CLI flag:      --gpus N   (takes priority over everything)
    • Env variable:  NGPU=N ./QUICK_START.sh train
    • Default:       4
    • Example:       ./QUICK_START.sh train --gpus 1

  USE_CACHE explained:
    • Set in YAML:   teacher.use_cache: true/false  (permanent for that config)
    • CLI override:  --use-cache flag                (one-off, OR with YAML)
    • Default:       false  (live teacher, best signal quality)
    • Cache mode:    faster training, slight KD signal mismatch on augmented data
  ─────────────────────────────────────────────────────────────────────────────
EOF
}

# ── main ─────────────────────────────────────────────────────────────────────

# Parse --gpus N from any position in the argument list
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            NGPU="$2"
            shift 2
            ;;
        --gpus=*)
            NGPU="${1#--gpus=}"
            shift
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${ARGS[@]}"

case "${1:-help}" in
    train)        do_train ;;
    smoke)        do_smoke ;;
    cache)        do_cache ;;
    train-cache)  do_train_cache ;;
    eval)         do_eval "${2:-}" "${3:-}" ;;
    track)        do_track "${2:-}" "${3:-}" ;;
    monitor)      do_monitor ;;
    export)       do_export ;;
    check)        check_training ;;
    help|--help|-h) do_help ;;
    *)
        echo "Unknown command: $1"
        do_help
        exit 1
        ;;
esac
