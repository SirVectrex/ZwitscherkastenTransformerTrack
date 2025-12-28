"""
Visualization script for training metrics.
Generates plots from saved metrics JSON files.

Usage:
    python visualize_metrics.py --run_name phase_2_20250228-143022
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_metrics(metrics_path):
    """Load metrics from JSON file."""
    with open(metrics_path, 'r') as f:
        data = json.load(f)
    return data


def plot_metrics(metrics_data, output_dir=None):
    """
    Create comprehensive visualization of training metrics.
    """
    epochs_data = metrics_data['epochs']
    
    if not epochs_data:
        print("No epoch data found!")
        return
    
    # Extract data
    epochs = [e['epoch'] for e in epochs_data]
    train_loss = [e['train_loss'] for e in epochs_data]
    val_loss = [e['val_loss'] for e in epochs_data]
    train_acc = [e['train_acc_top1'] for e in epochs_data]
    val_acc = [e['val_acc_top1'] for e in epochs_data]
    train_top5 = [e['train_acc_top5'] for e in epochs_data]
    val_top5 = [e['val_acc_top5'] for e in epochs_data]
    lrs = [e['learning_rate'] for e in epochs_data]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics Overview', fontsize=16, fontweight='bold')
    
    # 1. Loss Curve
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, 'o-', label='Train Loss', linewidth=2, markersize=4)
    ax.plot(epochs, val_loss, 's-', label='Val Loss', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Top-1 Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, train_acc, 'o-', label='Train Top-1', linewidth=2, markersize=4)
    ax.plot(epochs, val_acc, 's-', label='Val Top-1', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Top-1 Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # 3. Top-5 Accuracy
    ax = axes[1, 0]
    ax.plot(epochs, train_top5, 'o-', label='Train Top-5', linewidth=2, markersize=4)
    ax.plot(epochs, val_top5, 's-', label='Val Top-5', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Top-5 Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # 4. Learning Rate Schedule
    ax = axes[1, 1]
    ax.semilogy(epochs, lrs, 'o-', color='purple', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate (log scale)')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / 'metrics_overview.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved plot to {save_path}")
    else:
        plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("TRAINING SUMMARY STATISTICS")
    print("="*60)
    
    best_epoch_top1 = np.argmax(val_acc) + 1
    best_epoch_top5 = np.argmax(val_top5) + 1
    
    print(f"Best Val Top-1 Acc: {max(val_acc):.2f}% (Epoch {best_epoch_top1})")
    print(f"Best Val Top-5 Acc: {max(val_top5):.2f}% (Epoch {best_epoch_top5})")
    print(f"Final Val Top-1 Acc: {val_acc[-1]:.2f}%")
    print(f"Final Val Top-5 Acc: {val_top5[-1]:.2f}%")
    print(f"\nFinal Val Loss: {val_loss[-1]:.6f}")
    print(f"Final Train Loss: {train_loss[-1]:.6f}")
    print("="*60 + "\n")
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Visualize bird species classification training metrics'
    )
    parser.add_argument(
        '--metrics_file',
        type=str,
        help='Path to metrics JSON file (e.g., metrics/phase_2_20250228-143022_metrics.json)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='plots',
        help='Directory to save plots (default: ./plots)'
    )
    
    args = parser.parse_args()
    
    if not args.metrics_file:
        print("❌ Please provide --metrics_file argument")
        print("Example: python visualize_metrics.py --metrics_file metrics/phase_2_xxx_metrics.json")
        return
    
    metrics_path = Path(args.metrics_file)
    
    if not metrics_path.exists():
        print(f"❌ Metrics file not found: {metrics_path}")
        return
    
    print(f"📊 Loading metrics from {metrics_path}")
    metrics_data = load_metrics(metrics_path)
    
    # Print config info
    print(f"Phase: {metrics_data.get('phase', 'N/A')}")
    print(f"Focal Loss: {metrics_data.get('use_focal_loss', False)}")
    print(f"Total Epochs: {len(metrics_data['epochs'])}\n")
    
    # Generate plots
    plot_metrics(metrics_data, output_dir=args.output_dir)


if __name__ == "__main__":
    main()