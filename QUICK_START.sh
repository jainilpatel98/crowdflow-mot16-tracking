#!/bin/bash
# Quick Start Guide for Model Training

echo "================================================"
echo "MOT16 Person Tracking - Model Training Guide"
echo "================================================"
echo ""

# Check if training is running
check_training() {
    echo "🔍 Checking active training processes..."
    ps aux | grep -E "torchrun|train_student" | grep -v grep | wc -l
    echo ""
}

# Show available configs
show_configs() {
    echo "📋 Available Configurations:"
    echo ""
    echo "Full Training (60 epochs):"
    echo "  1. ResNet50          - Best accuracy, needs 4 GPUs"
    echo "  2. MobileNetV3-Large - Good balance, efficient"
    echo ""
    echo "Smoke Tests (2 epochs):"
    echo "  3. ResNet50 Test     - Quick validation"
    echo "  4. MobileNetV3-Large Test - Quick validation"
    echo ""
}

# Show resource requirements
show_resources() {
    echo "💾 GPU Memory Requirements:"
    echo ""
    echo "ResNet50 (12 batch/GPU):"
    echo "  - Per GPU: ~10GB (mixed precision)"
    echo "  - Total: ~40GB across 4 GPUs"
    echo ""
    echo "MobileNetV3-Large (14 batch/GPU):"
    echo "  - Per GPU: ~8GB (mixed precision)"
    echo "  - Total: ~32GB across 4 GPUs"
    echo ""
}

# Start training
start_training() {
    local config=$1
    echo "🚀 Starting training with: $config"
    cd /home/research/tracking_crowded_people
    source .venv/bin/activate
    torchrun --nproc_per_node=4 tools/train_student.py --config="$config"
}

# Show help
show_help() {
    echo "Usage: ./QUICK_START.sh [command]"
    echo ""
    echo "Commands:"
    echo "  check     - Check if training is running"
    echo "  configs   - Show available configurations"
    echo "  resources - Show GPU memory requirements"
    echo "  monitor   - Monitor training progress"
    echo "  help      - Show this help message"
    echo ""
}

# Main
case "${1:-help}" in
    check)
        check_training
        ;;
    configs)
        show_configs
        ;;
    resources)
        show_resources
        ;;
    monitor)
        cd /home/research/tracking_crowded_people
        source .venv/bin/activate
        python monitor_training.py runs/student_distill_resnet50
        ;;
    *)
        show_help
        show_configs
        echo "Current Training Status:"
        check_training
        ;;
esac
