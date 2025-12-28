#!/bin/bash
# setup_training.sh - Initialize all directories and check dependencies

set -e

echo "🚀 Setting up Bird Acoustics Training Pipeline..."
echo "================================================"

# Create required directories
mkdir -p checkpoints
mkdir -p logs
mkdir -p metrics
mkdir -p analysis_plots
mkdir -p plots

echo "✅ Created directories: checkpoints, logs, metrics, analysis_plots, plots"

# Check Python dependencies
echo ""
echo "Checking Python dependencies..."

python3 -c "import torch; print(f'✅ PyTorch {torch.__version__}')" || \
  (echo "❌ PyTorch not found. Install: pip install torch" && exit 1)

python3 -c "import pandas; print(f'✅ Pandas {pandas.__version__}')" || \
  (echo "❌ Pandas not found. Install: pip install pandas" && exit 1)

python3 -c "import numpy; print(f'✅ NumPy {numpy.__version__}')" || \
  (echo "❌ NumPy not found. Install: pip install numpy" && exit 1)

python3 -c "import sklearn; print(f'✅ Scikit-Learn {sklearn.__version__}')" || \
  (echo "❌ Scikit-Learn not found. Install: pip install scikit-learn" && exit 1)

python3 -c "import matplotlib; print(f'✅ Matplotlib {matplotlib.__version__}')" || \
  (echo "❌ Matplotlib not found. Install: pip install matplotlib" && exit 1)

python3 -c "from hear21passt.base import get_basic_model; print('✅ PaSST installed')" || \
  (echo "❌ PaSST not found. Install: pip install hear21passt" && exit 1)

echo ""
echo "================================================"
echo "✅ All setup checks passed!"
echo ""
echo "Next steps:"
echo "  1. python prepare_data.py    # Prepare and oversample data"
echo "  2. python analyze_dataset.py # Visualize class distribution"
echo "  3. python train.py           # Start training (set PHASE=1 first)"
echo "  4. python visualize_metrics.py --metrics_file <path> # View results"
echo ""
