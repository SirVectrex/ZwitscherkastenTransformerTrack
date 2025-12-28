## 🎓 COMPLETE TRAINING PIPELINE SUMMARY

Your bird acoustics classification system is now fully ready with **long-tail distribution handling**! Here's everything you need to know:

---

## 📦 FILES CREATED

### Core Training Files
1. **prepare_data.py** - Data preparation with stratified oversampling
2. **dataset.py** - PyTorch Dataset with SpecAugment
3. **train.py** - Full training loop with metrics tracking
4. **analyze_dataset.py** - Dataset statistics and visualization
5. **visualize_metrics.py** - Post-training metric visualization
6. **setup_training.sh** - Dependency checker

### Documentation
7. **README.md** - Complete guide and reference
8. **TRAINING_GUIDE.md** - This file

---

## 🚀 EXECUTION WORKFLOW

### Phase 0: Setup (One Time)
```bash
bash setup_training.sh
# Checks all dependencies and creates directories
```

### Phase 1: Data Preparation
```bash
python prepare_data.py
```

**What happens:**
- Scans your preprocessed .npy files
- **Oversamples** rare bird species to MIN_SAMPLES_PER_CLASS (default: 500)
- Computes inverse frequency weights for the loss function
- Splits into train (90%) / val (10%)
- Saves: `train.csv`, `val.csv`, `class_weights.json`, `class_map.json`, etc.

**Customization:**
```python
MIN_SAMPLES_PER_CLASS = 500  # Change this value
# Higher = more oversampling = more data duplication
# Lower = less duplication = faster training, less accuracy
```

### Phase 2: Analyze Data Distribution
```bash
python analyze_dataset.py
```

**Produces:**
- Console output with class statistics
- `analysis_plots/dataset_analysis.png` showing:
  - Bar chart of samples per class (after oversampling)
  - Class weights visualization
  - Log-scale distribution
  - Train/Val split pie chart

**Review this before training** to ensure oversampling is appropriate!

### Phase 3a: Train Phase 1 (Head Only)
```bash
# In train.py, set:
PHASE = 1

# Then run:
python train.py
```

**What this does:**
- Loads pre-trained PaSST backbone (frozen)
- Trains ONLY the classification head
- ~20 epochs, learning rate 1e-4
- Faster training, lower memory usage
- Saves best model to `checkpoints/best_model_phase_1.pth`

**Expected result:** ~85-88% accuracy (depending on data)

### Phase 3b: Train Phase 2 (Full Fine-Tuning)
```bash
# In train.py, set:
PHASE = 2

# Then run:
python train.py
```

**What this does:**
- Loads Phase 1 checkpoint automatically
- Fine-tunes ENTIRE model
- ~30 epochs, learning rate 1e-5 (lower, more careful)
- Mixup augmentation enabled (virtual training samples)
- SpecAugment still active
- Saves best model to `checkpoints/best_model_phase_2.pth`

**Expected result:** ~87-92% accuracy

---

## 📊 UNDERSTANDING THE LOSS FUNCTIONS

### Default: Class-Weighted CrossEntropyLoss

**How it works:**
```
Loss = -weight[class] * log(p_class)

Where weight[class] = num_classes / (count[class] * num_classes)
```

**Effect:** 
- Rare birds get high weights → bigger loss when misclassified
- Common birds get low weights → smaller loss (already well-learned)
- Trains in one pass, no data duplication (unlike oversampling)

**Example:**
```
Rare bird (20 samples): weight = 5x
Common bird (500 samples): weight = 1x

If both misclassified:
  Rare bird loss = 5x base loss
  Common bird loss = 1x base loss
```

### Optional: Focal Loss

**Enable with:**
```python
USE_FOCAL_LOSS = True
```

**How it works:**
```
Focal Loss = -alpha_t * (1 - p_t)^gamma * log(p_t)

Where:
  alpha_t = class weight (same as above)
  p_t = probability of true class (0-1)
  gamma = 2.0 (focusing parameter)
```

**Effect:**
- Easy examples (p_t near 1): loss ≈ 0 (ignored)
- Hard examples (p_t near 0): loss = full weight (focused)
- Combines class imbalance AND hard example handling

**Use Focal Loss when:**
- You still have >90% imbalance ratio after oversampling
- Regular loss plateaus
- You want to focus on "hard to learn" birds

---

## 📈 INTERPRETING METRICS

All metrics saved in two formats:

### 1. CSV Logs (`logs/phase_X_TIMESTAMP_metrics.csv`)
```
epoch | train_loss | train_acc | train_top5_acc | val_loss | val_acc | val_top5_acc | lr
```

- Simple, spreadsheet-friendly format
- Can open in Excel
- Useful for quick checks

### 2. JSON Metrics (`metrics/phase_X_TIMESTAMP_metrics.json`)
```json
{
  "config": {...},
  "phase": 2,
  "use_focal_loss": false,
  "epochs": [
    {
      "epoch": 1,
      "train_loss": 2.3456,
      "train_acc_top1": 45.23,
      "train_acc_top5": 78.90,
      "val_loss": 2.2341,
      "val_acc_top1": 47.12,
      "val_acc_top5": 81.23,
      "learning_rate": 1e-4
    },
    ...
  ]
}
```

- Complete detailed history
- Machine-readable for analysis
- Used by `visualize_metrics.py`

### 3. TensorBoard Logs (`logs/phase_X_TIMESTAMP/`)
```bash
tensorboard --logdir=logs/
# Then visit http://localhost:6006 in browser
```

- Real-time metric visualization
- Interactive graphs
- Multiple runs comparison

---

## 🎯 VISUALIZATION AFTER TRAINING

### Generate Comprehensive Plots
```bash
python visualize_metrics.py \
  --metrics_file metrics/phase_2_20250228-143022_metrics.json \
  --output_dir plots
```

**Output:** 4-panel figure with:
1. **Loss Curves**: Train vs Val loss over epochs
2. **Top-1 Accuracy**: Strict metric (1st prediction must be correct)
3. **Top-5 Accuracy**: Lenient metric (correct label in top-5)
4. **Learning Rate Schedule**: Shows LR changes during training

**Also prints:**
```
Best Val Top-1 Acc: 89.23% (Epoch 25)
Best Val Top-5 Acc: 96.45% (Epoch 24)
Final Val Top-1 Acc: 88.91%
Final Val Top-5 Acc: 96.12%
```

---

## 🔍 DEBUGGING & OPTIMIZATION

### If accuracy is stuck around 80%

**Try in order:**

1. **Check oversampling:**
   ```bash
   python analyze_dataset.py
   # Look at class imbalance ratio in output
   # If still > 5x after oversampling, increase MIN_SAMPLES_PER_CLASS
   ```

2. **Increase oversampling threshold:**
   ```python
   MIN_SAMPLES_PER_CLASS = 1000  # More aggressive
   python prepare_data.py
   python train.py  # Retrain
   ```

3. **Switch to Focal Loss:**
   ```python
   USE_FOCAL_LOSS = True
   python train.py  # Retrain Phase 2
   ```

4. **Train longer:**
   ```python
   "epochs": 50,  # Increase from 30
   ```

5. **Adjust learning rate:**
   ```python
   "lr": 2e-5,  # Try different values
   ```

### If accuracy is decreasing after N epochs

**Signs of overfitting:**
- Train loss keeps decreasing
- Val loss starts increasing
- Val accuracy plateaus or decreases

**Solutions:**
1. Enable early stopping (manual in code)
2. Increase label smoothing: `"label_smoothing": 0.2`
3. Increase SpecAugment strength:
   ```python
   freq_mask_param=25  # was 15
   time_mask_param=50  # was 35
   ```
4. Increase weight decay: `"weight_decay": 0.001`

### If training is too slow

**Memory-conscious optimizations:**
```python
"batch_size": 2,  # Reduce from 4 (slower, but fits)
# OR
accum_steps = 32  # Accumulate more gradients
```

**Check GPU usage:**
```bash
nvidia-smi -l 1  # Update every 1 second
# Should see ~80% memory utilization
```

---

## 📋 CONFIGURATION REFERENCE

### Key Hyperparameters to Tune

| Parameter | Current | Try Range | Effect |
|-----------|---------|-----------|--------|
| `MIN_SAMPLES_PER_CLASS` | 500 | 300-1000 | Oversampling aggressiveness |
| `batch_size` | 4 | 2-8 | Larger = faster, more VRAM |
| `epochs` | 30 | 20-50 | Longer training = potentially higher acc |
| `lr` (Phase 1) | 1e-4 | 5e-5 to 1e-3 | Higher = faster convergence, instability |
| `lr` (Phase 2) | 1e-5 | 5e-6 to 1e-4 | Fine-tune carefully |
| `label_smoothing` | 0.1 | 0-0.3 | Regularization strength |
| `MIXUP_ALPHA` | 0.4 | 0.2-0.8 | Mixup interpolation strength |
| `freq_mask_param` | 15 | 10-30 | SpecAugment freq masking |
| `time_mask_param` | 35 | 20-60 | SpecAugment time masking |

### Loss Function Comparison

| Loss | Pros | Cons | Best For |
|------|------|------|----------|
| **Weighted CE** | Simple, fast, effective | No hard-example focus | Moderate imbalance |
| **Focal Loss** | Focuses on hard examples | Slower, more hyperparams | Severe imbalance |
| **Weighted CE + Oversampling** | Balanced, interpretable | More data to train on | When you have enough data |

---

## ✅ CHECKLIST BEFORE SUBMITTING

Before considering your model complete:

- [ ] Run `prepare_data.py` → check `class_weights.json` looks reasonable
- [ ] Run `analyze_dataset.py` → verify oversampling is effective
- [ ] Train Phase 1 → reaches ~85% accuracy
- [ ] Train Phase 2 → reaches at least 88%+ accuracy
- [ ] Run `visualize_metrics.py` → loss curves look smooth (no spikes)
- [ ] Check top-5 accuracy → should be 5-10% higher than top-1
- [ ] Save best checkpoint → `best_model_phase_2.pth`
- [ ] Document your final accuracy in paper/thesis
- [ ] Save metrics JSON for reproducibility

---

## 🎓 THEORETICAL BACKGROUND

### Why Long-Tail Matters in Bird Classification

Real bird datasets are **long-tail**: many common species, few rare ones.

```
Common species (Robin, Sparrow):     10,000 samples
Moderate species (Warbler):          1,000 samples
Rare species (Whooping Crane):       50 samples
```

**Problem:** Standard training optimizes for **total accuracy**, not per-class.

```
Model learns: "Always predict Robin" = 85% accuracy ✓
But fails on rare species = 5% accuracy ✗
```

**Solution:** Weight by inverse frequency
```
weight[Robin] = 1.0     (common, low weight)
weight[Warbler] = 10.0  (moderate weight)
weight[Crane] = 200.0   (rare, high weight)

Now model can't just "cheat" with common species!
```

### Why Oversampling Helps

Oversampling brings all classes to ~same frequency:
```
Before: [10000, 1000, 50]     Long-tail distribution
After:  [500, 500, 500]       Balanced
```

**Benefits:**
- Model sees rare species more often
- Gradient updates more varied
- Better feature learning for minorities

**Drawback:**
- More duplicated data = potential overfitting
- Slower training

### Why Focal Loss Helps Even More

Focal Loss applies **dynamic weighting** based on **difficulty**:
```
Easy example (p_t=0.99): loss ≈ 0 (ignored)
Hard example (p_t=0.3):  loss = FULL WEIGHT (focused)
```

This ensures training focuses on:
1. **Hard classes** (rare species)
2. **Hard samples** (confusing instances)

---

## 📚 FURTHER READING

Implement these if you want to go deeper:

1. **Confusion Matrix**: See which species confuse the model
   ```python
   from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
   ```

2. **Per-Class Metrics**: Separate accuracy for each bird
   ```python
   from sklearn.metrics import classification_report
   ```

3. **Calibration**: Is model confidence aligned with accuracy?
   ```python
   from sklearn.calibration import calibration_curve
   ```

4. **Attention Visualization**: Which Mel frequencies matter most?
   ```python
   # Use GradCAM on PaSST attention maps
   ```

---

## 💡 FINAL TIPS

1. **Always check data first** before tweaking hyperparams
   ```bash
   python analyze_dataset.py  # Do this always
   ```

2. **Train incrementally**: Phase 1 → Phase 2 → More Phase 2
   Don't jump straight to 50 epochs Phase 2

3. **Save ALL checkpoints** during Phase 2
   ```python
   # Modify train.py to save every N epochs
   if (epoch + 1) % 5 == 0:
       torch.save(model.state_dict(), f"checkpoints/epoch_{epoch+1}.pth")
   ```

4. **Monitor Top-5 closely**
   - If Top-1: 85%, Top-5: 87% → model is confused
   - If Top-1: 85%, Top-5: 96% → model "knows" but ranks wrong

5. **Don't obsess over 2% difference**
   - 85% vs 87% might just be random seed
   - Run multiple times to get average performance

---

## 🆘 STILL STUCK?

Check in this order:
1. Dataset analysis output → class distribution making sense?
2. Loss curves from TensorBoard → smooth? Decreasing?
3. Validation accuracy → increasing? Plateauing?
4. Class weights JSON → rare classes have higher weights?

If all looks good but accuracy low:
- Your mel-spectrogram preprocessing might be the issue
- Check normalization: mean=-4.27, std=4.57
- Try different PaSST weights or models

---

**Good luck with your thesis! You've got a solid pipeline now. 🚀**

Questions? Check the README.md or look at the code comments!
