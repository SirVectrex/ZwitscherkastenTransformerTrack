#!/usr/bin/env python3
"""
QUICK START SCRIPT
Run this to set up and do a quick test of your pipeline.

Usage:
    python quickstart.py
"""

import subprocess
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).parent.resolve()


def run_command(cmd, description):
    """Run a shell command and report status."""
    print(f"\n{'='*70}")
    print(f"📍 {description}")
    print(f"{'='*70}")
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"✅ {description} completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error:")
        print(f"   {e}")
        return False


def main():
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                     BIRD ACOUSTICS PIPELINE                        ║
║                      Quick Start Guide                             ║
╚════════════════════════════════════════════════════════════════════╝
    """)
    
    print("""
This script will guide you through the complete pipeline:
    1. Setup & dependencies
    2. Data preparation
    3. Dataset analysis
    4. Training (you can skip this for now)
    5. Visualization

Press Enter to continue or Ctrl+C to exit...
    """)
    
    input()
    
    # Step 1: Setup
    print("\n🔧 STEP 1: Checking Dependencies")
    cmd = "python -c \"import torch, pandas, numpy, sklearn, matplotlib; from hear21passt.base import get_basic_model; print('✓ All dependencies OK')\""
    if not run_command(cmd, "Dependency check"):
        print("❌ Please install missing dependencies first!")
        print("   pip install torch pandas numpy scikit-learn matplotlib hear21passt")
        return
    
    # Step 2: Create directories
    print("\n📁 STEP 2: Creating Required Directories")
    for d in ["checkpoints", "logs", "metrics", "analysis_plots", "plots"]:
        Path(d).mkdir(exist_ok=True)
        print(f"   ✓ {d}/")
    
    # Step 3: Prepare data
    print("\n📊 STEP 3: Preparing Data with Oversampling")
    print("""
This will:
  - Scan your preprocessed .npy files
  - Oversample rare bird species to 500 samples each
  - Compute class weights for the loss function
  - Split into train/val
  
Continue? (y/n): """, end="")
    
    if input().lower() == 'y':
        if not run_command("python prepare_data.py", "Data preparation"):
            return
        print("\n✅ Generated files:")
        for f in ["train.csv", "val.csv", "class_weights.json", "class_map.json", "dataset_summary.json"]:
            if (CURRENT_DIR / f).exists():
                print(f"   ✓ {f}")
    else:
        print("   ⏭️  Skipping data preparation")
    
    # Step 4: Analyze data
    print("\n📈 STEP 4: Analyzing Dataset Distribution")
    print("""
This will create:
  - Console summary of class distribution
  - analysis_plots/dataset_analysis.png with 4 visualizations
  
Continue? (y/n): """, end="")
    
    if input().lower() == 'y':
        if not run_command("python analyze_dataset.py", "Dataset analysis"):
            return
    else:
        print("   ⏭️  Skipping analysis")
    
    # Step 5: Training prompt
    print("\n🚀 STEP 5: Training (Optional)")
    print("""
To train your model, you need to:

  Phase 1 (Head Only):
    - Edit train.py and set: PHASE = 1
    - Run: python train.py
    - Takes ~30 minutes, reaches ~85% accuracy
  
  Phase 2 (Full Fine-Tune):
    - Edit train.py and set: PHASE = 2
    - Run: python train.py
    - Takes ~1-2 hours, reaches ~88-92% accuracy

  Continue training now? (y/n): """, end="")
    
    if input().lower() == 'y':
        phase = input("\nSelect phase (1 or 2): ").strip()
        if phase not in ['1', '2']:
            print("Invalid phase!")
            return
        
        # Update train.py to set phase
        print(f"   ⚠️  Make sure train.py has PHASE = {phase}")
        print("   Run: python train.py")
    else:
        print("\n   ⏭️  Skipping training for now")
        print("""   
   When ready to train:
     1. Edit train.py: set PHASE = 1
     2. Run: python train.py
     3. Then change PHASE = 2 and run again
        """)
    
    # Step 6: Visualization
    print("\n📊 STEP 6: Visualize Results (After Training)")
    print("""
After training completes, visualize your metrics:

  python visualize_metrics.py \\
    --metrics_file metrics/<your_metrics_file>.json \\
    --output_dir plots

This creates:
  - plots/metrics_overview.png (4-panel comparison)
  - Console summary with best accuracies
    """)
    
    print("\n" + "="*70)
    print("✅ SETUP COMPLETE!")
    print("="*70)
    print("""
Next steps:
  1. Review analysis_plots/dataset_analysis.png
  2. Read TRAINING_GUIDE.md for detailed information
  3. Start training: edit train.py to set PHASE = 1, then run
  
Questions? Check:
  - README.md (complete reference)
  - TRAINING_GUIDE.md (step-by-step guide)
  - train.py, dataset.py (code comments)
    """)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Cancelled by user")
        sys.exit(0)
