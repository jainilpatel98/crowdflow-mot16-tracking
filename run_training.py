#!/usr/bin/env python3
"""
Model Training Runner - Simplified interface for starting training with different model architectures
"""
from pathlib import Path
import subprocess
import sys


def run_training(model_config: str, num_gpus: int = 4):
    """
    Launch distributed training on specified number of GPUs
    
    Args:
        model_config: Config file name (e.g., 'student_distill_resnet50.yaml')
        num_gpus: Number of GPUs to use (default 4)
    """
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "tools/train_student.py",
        f"--config=configs/{model_config}"
    ]
    
    print(f"\n{'='*70}")
    print(f"Starting training with config: {model_config}")
    print(f"Number of GPUs: {num_gpus}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    # Available configurations
    configs = {
        "1": ("ResNet50 (Full 60 epochs)", "student_distill_resnet50.yaml", 4),
        "2": ("MobileNetV3-Large (Full 60 epochs)", "student_distill_mobilenetv3_large.yaml", 4),
        "3": ("ResNet50 Smoke Test (2 epochs)", "student_distill_resnet50_smoke.yaml", 4),
        "4": ("MobileNetV3-Large Smoke Test (2 epochs)", "student_distill_mobilenetv3_large_smoke.yaml", 4),
    }
    
    if len(sys.argv) > 1:
        # Allow direct specification: python run_training.py 1
        choice = sys.argv[1]
        if choice in configs:
            name, config, gpus = configs[choice]
            print(f"Selected: {name}")
            exit_code = run_training(config, gpus)
            sys.exit(exit_code)
        else:
            print(f"Invalid choice: {choice}")
    
    # Interactive menu
    print("\nAvailable Training Configurations:")
    print("="*70)
    for key, (name, config, gpus) in configs.items():
        print(f"{key}. {name:<45} (GPUs: {gpus})")
    print("="*70)
    
    choice = input("\nSelect configuration (1-4): ").strip()
    if choice in configs:
        name, config, gpus = configs[choice]
        print(f"\nSelected: {name}")
        exit_code = run_training(config, gpus)
        sys.exit(exit_code)
    else:
        print(f"Invalid choice: {choice}")
        sys.exit(1)
