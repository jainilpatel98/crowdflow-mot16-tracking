#!/bin/bash
# =============================================================================
# QUICK_START.sh — MOT16 YOLO-to-Student Distillation Pipeline
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
#   train       Full student distillation run (default student = ResNet50)
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

_student_variant_key() {
    local variant="${1:-resnet50}"
    case "${variant,,}" in
        ""|default|resnet50) echo "resnet50" ;;
        resnet101) echo "resnet101" ;;
        resnext50|resnext-50|resnext50_32x4d|resnext50-32x4d) echo "resnext50" ;;
        resnext101|resnext-101|resnext101_32x8d|resnext101-32x8d) echo "resnext101" ;;
        se_resnet50|se-resnet50|seresnet50) echo "se_resnet50" ;;
        se_resnet101|se-resnet101|seresnet101) echo "se_resnet101" ;;
        *) return 1 ;;
    esac
}

_is_student_variant() {
    _student_variant_key "$1" >/dev/null 2>&1
}

_student_config() {
    case "$(_student_variant_key "$1")" in
        resnet50) echo "configs/student_distill_resnet50.yaml" ;;
        resnet101) echo "configs/student_distill_resnet101.yaml" ;;
        resnext50) echo "configs/student_distill_resnext50.yaml" ;;
        resnext101) echo "configs/student_distill_resnext101.yaml" ;;
        se_resnet50) echo "configs/student_distill_se_resnet50.yaml" ;;
        se_resnet101) echo "configs/student_distill_se_resnet101.yaml" ;;
    esac
}

_student_output_dir() {
    case "$(_student_variant_key "$1")" in
        resnet50) echo "runs/student_distill_resnet50" ;;
        resnet101) echo "runs/student_distill_resnet101" ;;
        resnext50) echo "runs/student_distill_resnext50" ;;
        resnext101) echo "runs/student_distill_resnext101" ;;
        se_resnet50) echo "runs/student_distill_se_resnet50" ;;
        se_resnet101) echo "runs/student_distill_se_resnet101" ;;
    esac
}

_student_variant_label() {
    case "$(_student_variant_key "$1")" in
        resnet50) echo "ResNet50" ;;
        resnet101) echo "ResNet101" ;;
        resnext50) echo "ResNeXt-50 (32x4d)" ;;
        resnext101) echo "ResNeXt-101 (32x8d)" ;;
        se_resnet50) echo "SE-ResNet-50" ;;
        se_resnet101) echo "SE-ResNet-101" ;;
    esac
}

_student_default_checkpoint() {
    local out_dir
    out_dir="$(_student_output_dir "$1")"
    echo "$out_dir/best.pt"
}

_student_export_path() {
    case "$(_student_variant_key "$1")" in
        resnet50) echo "runs/student_distill_resnet50/student_resnet50.onnx" ;;
        resnet101) echo "runs/student_distill_resnet101/student_resnet101.onnx" ;;
        resnext50) echo "runs/student_distill_resnext50/student_resnext50.onnx" ;;
        resnext101) echo "runs/student_distill_resnext101/student_resnext101.onnx" ;;
        se_resnet50) echo "runs/student_distill_se_resnet50/student_se_resnet50.onnx" ;;
        se_resnet101) echo "runs/student_distill_se_resnet101/student_se_resnet101.onnx" ;;
    esac
}

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
    local variant="${1:-resnet50}"
    local config
    config="$(_student_config "$variant")" || { echo "Unknown student variant: $variant"; exit 1; }
    local variant_label
    variant_label="$(_student_variant_label "$variant")"
    local output_dir
    output_dir="$(_student_output_dir "$variant")"
    _header "Full $variant_label Distillation Training (80 epochs, live teacher)"
    # ─── COMMAND ───────────────────────────────────────────────────────────
    # Default: live teacher — teacher and student share the same augmented input
    #   torchrun --nproc_per_node=4 tools/train_student.py \
    #       --config $config
    #
    # To resume from a checkpoint:
    #   torchrun --nproc_per_node=4 tools/train_student.py \
    #       --config $config \
    #       --resume $output_dir/last.pt
    #
    # To enable cache mode (faster, slight accuracy trade-off):
    #   torchrun --nproc_per_node=4 tools/train_student.py \
    #       --config $config \
    #       --use-cache
    # ───────────────────────────────────────────────────────────────────────
    cd "$ROOT"
    _cmd "torchrun --nproc_per_node=$NGPU tools/train_student.py --config $config"
    torchrun --nproc_per_node="$NGPU" tools/train_student.py \
        --config "$config"
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
    local variant="${1:-resnet50}"
    local config
    config="$(_student_config "$variant")" || { echo "Unknown student variant: $variant"; exit 1; }
    local variant_label
    variant_label="$(_student_variant_label "$variant")"
    _header "Pre-compute Teacher Cache  (optional, for cache-mode training)"
    echo "  NOTE: Default training is live-teacher (no cache needed)."
    echo "  Use cache only if GPU memory is constrained or teacher is slow."
    echo "  Student variant: $variant_label"
    echo ""
    # ─── COMMANDS ──────────────────────────────────────────────────────────
    # Step 1: cache train split
    #   python tools/cache_teacher_outputs.py \
    #       --config $config --split train
    #
    # Step 2: cache val split
    #   python tools/cache_teacher_outputs.py \
    #       --config $config --split val
    #
    # Step 3: train with cache (--use-cache flag)
    #   torchrun --nproc_per_node=4 tools/train_student.py \
    #       --config $config --use-cache
    # ───────────────────────────────────────────────────────────────────────
    cd "$ROOT"
    _cmd "python tools/cache_teacher_outputs.py --config $config --split train"
    "$PYTHON" tools/cache_teacher_outputs.py \
        --config "$config" --split train

    _cmd "python tools/cache_teacher_outputs.py --config $config --split val"
    "$PYTHON" tools/cache_teacher_outputs.py \
        --config "$config" --split val

    echo ""
    echo "  Cache written to: data/teacher_cache/"
    echo "  Now run:  ./QUICK_START.sh train-cache   (or pass --use-cache manually)"
}

do_train_cache() {
    local variant="${1:-resnet50}"
    local config
    config="$(_student_config "$variant")" || { echo "Unknown student variant: $variant"; exit 1; }
    local variant_label
    variant_label="$(_student_variant_label "$variant")"
    _header "Train $variant_label with Pre-computed Cache"
    # ─── COMMAND ───────────────────────────────────────────────────────────
    #   torchrun --nproc_per_node=4 tools/train_student.py \
    #       --config $config \
    #       --use-cache
    # ───────────────────────────────────────────────────────────────────────
    cd "$ROOT"
    _cmd "torchrun --nproc_per_node=$NGPU tools/train_student.py --config $config --use-cache"
    torchrun --nproc_per_node="$NGPU" tools/train_student.py \
        --config "$config" \
        --use-cache
}

do_eval() {
    local model_target="${1:-}"
    cd "$ROOT"
    if [ "$model_target" = "teacher" ]; then
        local variant="resnet50"
        local teacher_ckpt=""
        if _is_student_variant "${2:-}"; then
            variant="$(_student_variant_key "$2")"
            teacher_ckpt="${3:-}"
        else
            teacher_ckpt="${2:-}"
        fi
        local config
        config="$(_student_config "$variant")"
        _header "Evaluate Teacher Detection on MOT16 Val Sequences"
        if [ -n "$teacher_ckpt" ]; then
            _cmd "python tools/eval_detection.py --config $config --model-type teacher --checkpoint $teacher_ckpt"
            "$PYTHON" tools/eval_detection.py \
                --config "$config" \
                --model-type teacher \
                --checkpoint "$teacher_ckpt"
        else
            _cmd "python tools/eval_detection.py --config $config --model-type teacher"
            "$PYTHON" tools/eval_detection.py \
                --config "$config" \
                --model-type teacher
        fi
    else
        local variant="resnet50"
        local ckpt=""
        if _is_student_variant "$model_target"; then
            variant="$(_student_variant_key "$model_target")"
            ckpt="${2:-$(_student_default_checkpoint "$variant")}"
        else
            ckpt="${1:-$(_student_default_checkpoint "$variant")}"
        fi
        local config
        config="$(_student_config "$variant")"
        local variant_label
        variant_label="$(_student_variant_label "$variant")"
        # ─── COMMAND ───────────────────────────────────────────────────────
        #   python tools/eval_detection.py \
        #       --config $config \
        #       --checkpoint $ckpt
        # ───────────────────────────────────────────────────────────────────
        _header "Evaluate $variant_label Detection on MOT16 Val Sequences"
        _cmd "python tools/eval_detection.py --config $config --checkpoint $ckpt"
        "$PYTHON" tools/eval_detection.py \
            --config "$config" \
            --checkpoint "$ckpt"
    fi
}

do_track() {
    local variant="resnet50"
    local seq_arg="${1:-}"
    local ckpt_arg="${2:-}"
    if _is_student_variant "$seq_arg"; then
        variant="$(_student_variant_key "$seq_arg")"
        seq_arg="${2:-}"
        ckpt_arg="${3:-}"
    fi
    local config
    config="$(_student_config "$variant")" || { echo "Unknown student variant: $variant"; exit 1; }
    local variant_label
    variant_label="$(_student_variant_label "$variant")"
    _header "Run $variant_label Tracking on a MOT16 Sequence"
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
    local seq="${seq_arg:-MOT16/train/MOT16-10}"
    local out_dir="outputs/student_tracking"
    if [ "$variant" != "resnet50" ]; then
        out_dir="outputs/student_tracking_${variant}"
    fi
    local out="$out_dir/$(basename "$seq").txt"
    local out_video="$out_dir/$(basename "$seq").mp4"
    local ckpt="${ckpt_arg:-$(_student_default_checkpoint "$variant")}"
    cd "$ROOT"
    mkdir -p "$(dirname "$out")"
    _cmd "python tools/eval_tracking.py --sequence-dir $seq --output $out --output-video $out_video"
    "$PYTHON" tools/eval_tracking.py \
        --config "$config" \
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
    local variant="${1:-resnet50}"
    local output_dir
    output_dir="$(_student_output_dir "$variant")" || { echo "Unknown student variant: $variant"; exit 1; }
    local variant_label
    variant_label="$(_student_variant_label "$variant")"
    _header "Monitor $variant_label Training (live metrics)"
    # ─── COMMAND ───────────────────────────────────────────────────────────
    #   python monitor_training.py runs/student_distill_resnet50
    # ───────────────────────────────────────────────────────────────────────
    cd "$ROOT"
    "$PYTHON" monitor_training.py "$output_dir"
}

do_export() {
    local variant="${1:-resnet50}"
    local config
    config="$(_student_config "$variant")" || { echo "Unknown student variant: $variant"; exit 1; }
    local ckpt
    ckpt="$(_student_default_checkpoint "$variant")"
    local output_path
    output_path="$(_student_export_path "$variant")"
    local variant_label
    variant_label="$(_student_variant_label "$variant")"
    _header "Export $variant_label Best Checkpoint to ONNX"
    # ─── COMMAND ───────────────────────────────────────────────────────────
    #   python tools/export_onnx.py \
    #       --config $config \
    #       --checkpoint $ckpt \
    #       --output $output_path
    # ───────────────────────────────────────────────────────────────────────
    cd "$ROOT"
    _cmd "python tools/export_onnx.py --checkpoint $ckpt"
    "$PYTHON" tools/export_onnx.py \
        --config "$config" \
        --checkpoint "$ckpt" \
        --output "$output_path"
}

do_help() {
    _header "MOT16 Distillation Pipeline — Command Reference"
    cat <<'EOF'
  TRAINING
  ─────────────────────────────────────────────────────────────────────────────
  Full training (80 epochs, live teacher — default student = ResNet50):
    ./QUICK_START.sh train
    ./QUICK_START.sh train resnext101
    # supported variants:
    #   resnet50, resnet101, resnext50, resnext101, se_resnet50, se_resnet101
    # or directly:
    torchrun --nproc_per_node=4 tools/train_student.py \
        --config configs/student_distill_resnet50.yaml
    torchrun --nproc_per_node=4 tools/train_student.py \
        --config configs/student_distill_resnext101.yaml

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
    ./QUICK_START.sh eval resnext101
    # or:
    python tools/eval_detection.py \
        --config configs/student_distill_resnet50.yaml \
        --checkpoint runs/student_distill_resnet50/best.pt
    python tools/eval_detection.py \
        --config configs/student_distill_resnext101.yaml \
        --checkpoint runs/student_distill_resnext101/best.pt

  Teacher detection metrics:
    ./QUICK_START.sh eval teacher
    # or:
    python tools/eval_detection.py \
        --config configs/student_distill_resnet50.yaml \
        --model-type teacher

  Tracking on a single sequence (MOT format output):
    ./QUICK_START.sh track MOT16/train/MOT16-10
    ./QUICK_START.sh track resnext101 MOT16/train/MOT16-10
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
    ./QUICK_START.sh export resnext101
    # or:
    python tools/export_onnx.py \
        --config configs/student_distill_resnet50.yaml \
        --checkpoint runs/student_distill_resnet50/best.pt \
        --output runs/student_distill_resnet50/student_resnet50.onnx

  UTILITIES
  ─────────────────────────────────────────────────────────────────────────────
  Monitor live training metrics:
    ./QUICK_START.sh monitor
    ./QUICK_START.sh monitor resnext101
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
    train)        do_train "${2:-resnet50}" ;;
    smoke)        do_smoke ;;
    cache)        do_cache "${2:-resnet50}" ;;
    train-cache)  do_train_cache "${2:-resnet50}" ;;
    eval)         do_eval "${2:-}" "${3:-}" "${4:-}" ;;
    track)        do_track "${2:-}" "${3:-}" "${4:-}" ;;
    monitor)      do_monitor "${2:-resnet50}" ;;
    export)       do_export "${2:-resnet50}" ;;
    check)        check_training ;;
    help|--help|-h) do_help ;;
    *)
        echo "Unknown command: $1"
        do_help
        exit 1
        ;;
esac
