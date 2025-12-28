"""
Utility script to analyze dataset class distribution and weights.
Shows before/after oversampling statistics.

Usage:
    python analyze_dataset.py
"""

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def analyze_dataset():
    """Analyze and visualize dataset statistics."""
    
    CURRENT_DIR = Path(__file__).parent.resolve()
    
    # Load data
    try:
        with open(CURRENT_DIR / "dataset_summary.json") as f:
            summary = json.load(f)
        
        with open(CURRENT_DIR / "class_weights.json") as f:
            weights = json.load(f)
        
        with open(CURRENT_DIR / "idx_to_class.json") as f:
            idx_to_class = json.load(f)
        
        train_df = pd.read_csv(CURRENT_DIR / "train.csv")
        val_df = pd.read_csv(CURRENT_DIR / "val.csv")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run prepare_data.py first!")
        return
    
    # Print summary
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    
    print(f"\nTotal Classes: {summary['num_classes']}")
    print(f"Total Samples (after oversampling): {summary['total_samples']}")
    print(f"  Train samples: {summary['train_samples']} ({100*summary['train_samples']/summary['total_samples']:.1f}%)")
    print(f"  Val samples: {summary['val_samples']} ({100*summary['val_samples']/summary['total_samples']:.1f}%)")
    print(f"\nOversampling threshold: {summary['min_samples_per_class_after_oversampling']} samples/class")
    
    # Class distribution
    print("\n" + "="*70)
    print("CLASS DISTRIBUTION (AFTER OVERSAMPLING)")
    print("="*70)
    print(f"\n{'Class Name':<30} {'Train Samples':<15} {'Class Weight':<15}")
    print("-" * 60)
    
    class_counts = train_df['bird_name'].value_counts().sort_index()
    
    for idx in sorted([int(k) for k in weights.keys()]):
        class_name = idx_to_class[str(idx)]
        count = class_counts.get(class_name, 0)
        weight = weights[str(idx)]
        print(f"{class_name:<30} {count:<15} {weight:<15.4f}")
    
    # Statistics
    print("\n" + "="*70)
    print("CLASS DISTRIBUTION STATISTICS")
    print("="*70)
    
    counts = list(class_counts.values)
    print(f"\nMin samples/class: {min(counts)}")
    print(f"Max samples/class: {max(counts)}")
    print(f"Mean samples/class: {sum(counts) / len(counts):.1f}")
    print(f"Std Dev: {(sum((x - sum(counts)/len(counts))**2 for x in counts) / len(counts))**0.5:.1f}")
    
    # Class imbalance ratio
    imbalance_ratio = max(counts) / min(counts)
    print(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}x")
    print(f"  (Max samples / Min samples)")
    
    # Weight statistics
    weight_values = list(weights.values())
    print(f"\nWeight range: {min(weight_values):.4f} to {max(weight_values):.4f}")
    print(f"Weight ratio: {max(weight_values) / min(weight_values):.2f}x")
    print(f"  (Rarest class / Most common class weight)")
    
    # Visualize
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS...")
    print("="*70 + "\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Dataset Analysis', fontsize=16, fontweight='bold')
    
    # 1. Class distribution bar plot
    ax = axes[0, 0]
    class_names = [idx_to_class[str(i)] for i in sorted([int(k) for k in weights.keys()])]
    train_counts = [class_counts.get(name, 0) for name in class_names]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(class_names)))
    ax.bar(range(len(class_names)), train_counts, color=colors)
    ax.set_xlabel('Class Index')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Training Samples per Class (After Oversampling)')
    ax.tick_params(axis='x', rotation=45)
    
    # Add threshold line
    threshold = summary['min_samples_per_class_after_oversampling']
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Class weights
    ax = axes[0, 1]
    weight_values = [weights[str(i)] for i in sorted([int(k) for k in weights.keys()])]
    ax.bar(range(len(weight_values)), weight_values, color=colors)
    ax.set_xlabel('Class Index')
    ax.set_ylabel('Loss Weight')
    ax.set_title('Class Weights (for loss function)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Log-scale distribution
    ax = axes[1, 0]
    ax.bar(range(len(train_counts)), train_counts, color=colors)
    ax.set_yscale('log')
    ax.set_xlabel('Class Index')
    ax.set_ylabel('Number of Samples (log scale)')
    ax.set_title('Training Distribution (Log Scale)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y', which='both')
    
    # 4. Train/Val split
    ax = axes[1, 1]
    splits = ['Train', 'Validation']
    split_counts = [summary['train_samples'], summary['val_samples']]
    colors_split = ['#1f77b4', '#ff7f0e']
    
    wedges, texts, autotexts = ax.pie(
        split_counts, 
        labels=splits, 
        autopct='%1.1f%%',
        colors=colors_split,
        startangle=90,
        textprops={'fontsize': 12}
    )
    ax.set_title('Train/Val Split')
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = CURRENT_DIR / "analysis_plots"
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / "dataset_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved visualization to {save_path}")
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    import numpy as np
    analyze_dataset()