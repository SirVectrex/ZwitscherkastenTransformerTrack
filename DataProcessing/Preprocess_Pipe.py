import librosa
import numpy as np
import os
import shutil
import sys
import contextlib
import random
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

# ==========================================
# CONFIGURATION
# ==========================================

# PATHS
INPUT_RAW_ROOT = Path("./audio_data")            # Where your original MP3s are
INTERMEDIATE_ROOT = Path("./mel_pool_uint8")     # Temp folder for ALL processed specs
FINAL_OUTPUT_ROOT = Path("./split_dataset_1000_preprocessed") # Final destination for balanced data

# AUDIO PARAMETERS (PaSST Standard)
TARGET_SR = 16000           # 16kHz for PaSST
TARGET_DURATION = 10.0      # Seconds
N_MELS = 128
HOP_LENGTH = 160            # 10ms hop -> 1000 frames for 10s
WIN_LENGTH = 400            # 25ms window
FMIN = 50
FMAX = 8000

# PREPROCESSING INTELLIGENCE
INTENT_FILTER_MIN_HZ = 1500 # High-pass cutoff for detection (ignore wind)
INTENT_THRESHOLD_DB = 25    # dB relative to peak to consider "active"
MAX_SNIPPETS_PER_FILE = 5   # Extract at most this many good clips per MP3

# QUANTIZATION (0-255 uint8)
MIN_DB = -80.0
MAX_DB = 0.0

# BALANCING
TARGET_SAMPLES_PER_CLASS = 1000
NUM_SUBSETS = 5             # Number of cross-validation folds

# WORKERS
NUM_WORKERS = 8             # Adjust based on CPU cores

# ==========================================
# PART 1: AUDIO PREPROCESSING ENGINE
# ==========================================

@contextlib.contextmanager
def ignore_stderr():
    """Suppress librosa/audioread warnings."""
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        sys.stderr.flush()
        os.dup2(devnull, 2)
        os.close(devnull)
        yield
    except Exception:
        yield
    finally:
        try:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
        except Exception:
            pass

def process_single_mp3(file_path):
    """
    Reads one MP3, applies high-pass + intent detection, 
    loops short clips, creates Mel specs, quantizes to uint8.
    """
    try:
        with ignore_stderr():
            y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
    except Exception:
        return None, []

    if len(y) == 0: return None, []

    # --- A. Intent Detection ---
    # 1. Create a high-passed version just for detection (removes rumble)
    y_detect = librosa.effects.preemphasis(y, coef=0.97) 
    # (Simple preemphasis is often enough, or use butter_highpass if needed)
    
    # 2. Find non-silent intervals
    intervals = librosa.effects.split(y_detect, top_db=INTENT_THRESHOLD_DB)
    
    # Fallback: If no bird found, try to use the start of the file
    if len(intervals) == 0:
        intervals = np.array([[0, min(len(y), int(TARGET_SR * TARGET_DURATION))]])

    # --- B. Extraction & Processing ---
    specs = []
    target_samples = int(TARGET_SR * TARGET_DURATION)

    # Sort intervals by duration (longest/best first) to prioritize good data
    # (Optional, but often helps get the main song)
    intervals = sorted(intervals, key=lambda x: x[1]-x[0], reverse=True)

    for start, end in intervals:
        if len(specs) >= MAX_SNIPPETS_PER_FILE: break
        
        chunk = y[start:end]
        
        # 3. Looping / Padding
        if len(chunk) < target_samples:
            n_repeats = int(np.ceil(target_samples / len(chunk)))
            chunk = np.tile(chunk, n_repeats)[:target_samples]
        else:
            chunk = chunk[:target_samples]

        # 4. Audio Peak Normalization (Crucial for consistent volume)
        chunk = librosa.util.normalize(chunk)

        # 5. Mel Spectrogram Generation
        mel = librosa.feature.melspectrogram(
            y=chunk, sr=TARGET_SR, n_fft=1024, hop_length=HOP_LENGTH, 
            win_length=WIN_LENGTH, n_mels=N_MELS, fmin=FMIN, fmax=FMAX
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)

        # 6. Quantization to uint8
        log_mel = np.clip(log_mel, MIN_DB, MAX_DB)
        log_mel = (log_mel - MIN_DB) / (MAX_DB - MIN_DB) # 0.0 to 1.0
        
        # Ensure exact shape (128, 1000)
        if log_mel.shape[1] > 1000: log_mel = log_mel[:, :1000]
        if log_mel.shape[1] < 1000: 
            # Rare padding case
            log_mel = np.pad(log_mel, ((0,0), (0, 1000 - log_mel.shape[1])))
            
        quantized = (log_mel * 255).astype(np.uint8)
        specs.append(quantized)

    return file_path.parent.name, specs # Returns (class_name, list_of_specs)

def worker_wrapper(args):
    """Helper to unpack arguments for the process pool."""
    file_path, rel_path = args
    class_name, specs = process_single_mp3(file_path)
    
    if not specs: return "Failed"

    # Save immediately to INTERMEDIATE_ROOT
    # Structure: INTERMEDIATE_ROOT / ClassName / FileStem_i.npy
    save_dir = INTERMEDIATE_ROOT / class_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    stem = file_path.stem
    for i, spec in enumerate(specs):
        save_path = save_dir / f"{stem}_{i}.npy"
        np.save(save_path, spec)
        
    return "Success"

# ==========================================
# PART 2: BALANCING & SPLITTING ENGINE
# ==========================================

def balance_and_split():
    print("\n" + "="*50)
    print("STAGE 2: Balancing and Splitting Dataset")
    print("="*50)
    
    if not INTERMEDIATE_ROOT.exists():
        print(f"Error: Intermediate folder {INTERMEDIATE_ROOT} not found.")
        return

    classes = sorted([d.name for d in INTERMEDIATE_ROOT.iterdir() if d.is_dir()])
    valid_classes = []
    
    print("Scanning processed classes...")
    
    for cls in classes:
        cls_dir = INTERMEDIATE_ROOT / cls
        files = list(cls_dir.glob("*.npy"))
        count = len(files)
        
        if count >= TARGET_SAMPLES_PER_CLASS:
            valid_classes.append(cls)
        else:
            print(f"Skipping '{cls}': {count} samples (Required: {TARGET_SAMPLES_PER_CLASS})")

    if not valid_classes:
        print("CRITICAL: No classes met the sample requirement!")
        return

    imgs_per_subset = TARGET_SAMPLES_PER_CLASS // NUM_SUBSETS
    print(f"\nProceeding with {len(valid_classes)} classes.")
    print(f"Selecting {TARGET_SAMPLES_PER_CLASS} samples per class.")
    print(f"Splitting into {NUM_SUBSETS} folders ({imgs_per_subset} images each).")

    for cls in tqdm(valid_classes, desc="Distributing Files"):
        src_dir = INTERMEDIATE_ROOT / cls
        files = list(src_dir.glob("*.npy"))
        
        # Shuffle and pick exactly target amount
        random.shuffle(files)
        selected_files = files[:TARGET_SAMPLES_PER_CLASS]
        
        # Distribute into subsets
        for i in range(NUM_SUBSETS):
            subset_name = f"subset_{i}"
            dest_dir = FINAL_OUTPUT_ROOT / subset_name / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            start = i * imgs_per_subset
            end = start + imgs_per_subset
            batch = selected_files[start:end]
            
            for f in batch:
                shutil.copy2(f, dest_dir / f.name)

    print("\nDONE! Dataset is ready at:", FINAL_OUTPUT_ROOT)
    print(f"Structure: {FINAL_OUTPUT_ROOT}/subset_X/SpeciesName/file.npy")

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # --- STEP 1: PREPROCESSING ---
    print("="*50)
    print("STAGE 1: Audio Preprocessing & Feature Extraction")
    print("="*50)
    
    mp3_files = list(INPUT_RAW_ROOT.rglob("*.mp3"))
    if not mp3_files:
        print(f"No .mp3 files found in {INPUT_RAW_ROOT}")
        sys.exit()

    print(f"Found {len(mp3_files)} audio files. Starting workers...")
    
    # Prepare arguments for workers
    job_args = [(f, f.relative_to(INPUT_RAW_ROOT)) for f in mp3_files]
    
    # Run parallel processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        list(tqdm(executor.map(worker_wrapper, job_args), total=len(job_args), unit="file"))
        
    print("\nPreprocessing complete. Intermediate files saved.")

    # --- STEP 2: DOWNSAMPLING ---
    balance_and_split()