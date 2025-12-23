# SubSampled Data Track

## 🚀 Core Strategies

### 1. Intent-Triggered Extraction
Xeno-canto recordings often contain long periods of silence or wind noise. Our pipeline uses an **Energy-Based Intent Detector**:
* **High-Pass Filtering:** Pre-filters audio at **1.5kHz** to ignore low-frequency rumble during detection.
* **Activity Splitting:** Uses `librosa.effects.split` to identify actual bird vocalizations.
* **Smart Looping:** If a detected bird "intent" is shorter than the 10s target, the segment is **seamlessly looped** to provide a dense, information-rich sample for the Transformer attention layers.

### 2. Feature Engineering & Quantization
* **Standardized Input:** All audio is resampled to **16kHz** with a **160 hop length**, resulting in a 128 x 1000 Mel Spectrogram (exactly 10 seconds).
* **Storage Efficiency:** Spectrograms are saved as **8-bit integers (uint8)**. This reduces the dataset footprint by ~75% compared to float32, enabling faster disk I/O during training.
* **PaSST Alignment:** We apply **Z-Score Standardization** using AudioSet global statistics (Mean: -4.27, Std: 4.57) to ensure input features align with the pre-trained Transformer weights.

### 3. Overfitting Prevention (Regularization)
Transformers are prone to overfitting on small datasets. We mitigate this via:
* **Mixup Augmentation:** Blending two random samples and their labels during training.
* **SpecAugment:** Randomly masking frequency and time strips to force the model to learn context rather than specific "pixels".
* **Label Smoothing:** Prevents the model from becoming over-confident on noisy labels.
* **OneCycleLR:** Employs a learning rate "warm-up" and "cool-down" phase for stable convergence.