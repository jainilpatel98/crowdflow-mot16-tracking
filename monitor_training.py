#!/usr/bin/env python3
"""
Training Monitor - Real-time monitoring of distributed training progress
"""
import json
import sys
from pathlib import Path
from datetime import datetime


def monitor_training(output_dir: str = "runs/student_distill_resnet50"):
    """Monitor training progress from history.json"""
    output_path = Path(output_dir)
    history_file = output_path / "history.json"
    
    if not history_file.exists():
        print(f"❌ History file not found: {history_file}")
        print(f"   Training may not have started yet.")
        return
    
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        if not history:
            print("📊 Training just started, no history yet.")
            return
        
        latest = history[-1]
        epoch = latest['epoch']
        
        print(f"\n{'='*80}")
        print(f"📊 Training Progress Monitor - {output_dir}")
        print(f"{'='*80}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Current Epoch: {epoch}")
        
        # Training metrics
        train = latest['train']
        print(f"\n✅ Training Metrics (Epoch {epoch}):")
        print(f"  Total Loss:     {train['loss']:.6f}")
        print(f"  Detection:      {train['det']:.6f}")
        print(f"  Cls KD:         {train['cls_kd']:.6f}")
        print(f"  Box KD:         {train['box_kd']:.6f}")
        print(f"  Feature KD:     {train['feat_kd']:.6f}")
        print(f"  Embedding KD:   {train['emb_kd']:.6f}")
        print(f"  ID Loss:        {train['id_loss']:.6f}")
        
        # Validation metrics
        val = latest['val']
        print(f"\n🔍 Validation Metrics (Epoch {epoch}):")
        print(f"  Total Loss:     {val['loss']:.6f}")
        print(f"  Detection:      {val['det']:.6f}")
        print(f"  Cls KD:         {val['cls_kd']:.6f}")
        print(f"  Box KD:         {val['box_kd']:.6f}")
        print(f"  Feature KD:     {val['feat_kd']:.6f}")
        print(f"  Embedding KD:   {val['emb_kd']:.6f}")
        print(f"  ID Loss:        {val['id_loss']:.6f}")
        
        # Training progress
        if len(history) > 1:
            prev_loss = history[-2]['train']['loss']
            loss_delta = train['loss'] - prev_loss
            delta_str = f"({loss_delta:+.6f})"
            print(f"\n📈 Progress:")
            print(f"  Epochs completed: {epoch}/{60}")
            print(f"  Loss trend: {delta_str}")
            print(f"  Progress: [{epoch}/60 epochs]")
        
        # Best model info
        if len(history) > 1:
            best_val_loss = min(h['val']['loss'] for h in history)
            best_epoch = [h['epoch'] for h in history if h['val']['loss'] == best_val_loss][0]
            print(f"\n🏆 Best Model:")
            print(f"  Best epoch: {best_epoch}")
            print(f"  Best val loss: {best_val_loss:.6f}")
            print(f"  File: {output_path}/best.pt")
        
        print(f"\n{'='*80}\n")
        
    except Exception as e:
        print(f"❌ Error reading history: {e}")


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "runs/student_distill_resnet50"
    monitor_training(output_dir)
