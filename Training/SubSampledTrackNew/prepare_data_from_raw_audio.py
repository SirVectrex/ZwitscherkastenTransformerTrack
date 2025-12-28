"""
Preprocess Bird Audio Data from Raw MP3s - AUTO-DETECT PATH VERSION

This script automatically finds your audio_data folder by searching up the directory tree.
"""

import os
import librosa
import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


def find_audio_root():
    """
    Automatically find the audio_data folder by searching up the directory tree.
    Returns the path to audio_data/ folder.
    """
    current = Path(__file__).parent.resolve()
    
    print(f"🔍 Searching for audio_data folder...\n")
    print(f"Starting from: {current}\n")
    
    # Search up to 5 levels
    for i in range(6):
        search_path = current / "/".join([".."] * i) / "audio_data"
        search_path = search_path.resolve()
        
        print(f"  Checking: {search_path}")
        
        if search_path.exists():
            species = [d for d in search_path.iterdir() if d.is_dir()]
            if species:
                print(f"  ✅ Found! Contains {len(species)} species folders\n")
                return search_path
    
    # If not found automatically, show what we found
    print(f"\n❌ Could not find audio_data folder automatically.")
    print(f"\nLet's explore your directory structure:\n")
    
    # Show current directory and parent
    print(f"Current script location: {current}")
    print(f"Contents:\n")
    for item in sorted(current.parent.iterdir()):
        if item.is_dir():
            print(f"  📁 {item.name}/")
        else:
            print(f"  📄 {item.name}")
    
    print(f"\nParent directory ({current.parent.parent}):")
    for item in sorted(current.parent.parent.iterdir()):
        if item.is_dir():
            audio_files = list(item.glob("**/*.mp3")) + list(item.glob("**/*.wav"))
            if audio_files:
                print(f"  📁 {item.name}/ (has {len(audio_files)} audio files)")
            else:
                print(f"  📁 {item.name}/")
    
    return None


# --- CONFIGURATION ---
CURRENT_DIR = Path(__file__).parent.resolve()

# Auto-detect audio root
AUDIO_ROOT = find_audio_root()

if AUDIO_ROOT is None:
    print("\n" + "="*70)
    print("MANUAL FIX NEEDED")
    print("="*70)
    print("\nPlease edit this script and set AUDIO_ROOT manually to your audio_data path:")
    print("\nExample:")
    print("  AUDIO_ROOT = Path('/home/q490916/Documents/.../DataProcessing/audio_data')")
    exit(1)

PREPROCESSED_OUTPUT = CURRENT_DIR / "preprocessed_mels"

# Audio preprocessing
TARGET_SR = 32000
N_MELS = 128
N_FFT = 512
HOP_LENGTH = 512

# Dataset config
VAL_SIZE = 0.1
MIN_SAMPLES_PER_CLASS = 500
SEED = 42

print(f"="*70)
print(f"Using audio data from: {AUDIO_ROOT}")
print(f"="*70 + "\n")


def preprocess_audio_to_mel(audio_path, target_sr=32000, n_mels=128, n_fft=512):
    """
    Convert MP3/WAV → Mel spectrogram → uint8 (quantized)
    """
    try:
        y, sr = librosa.load(str(audio_path), sr=target_sr)
        
        mel = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=n_mels, 
            n_fft=n_fft,
            hop_length=HOP_LENGTH
        )
        
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = np.clip(mel_db, -80, 0)
        mel_uint8 = ((mel_db + 80) / 80 * 255).astype(np.uint8)
        
        return mel_uint8
    
    except Exception as e:
        print(f"  ⚠️  Error loading {audio_path.name}: {e}")
        return None


def main():
    print(f"\n{'='*70}")
    print("🎵 PREPROCESSING RAW AUDIO DATA")
    print(f"{'='*70}\n")
    
    # Scan bird species
    bird_species = sorted([
        d for d in AUDIO_ROOT.iterdir() 
        if d.is_dir()
    ])
    
    if not bird_species:
        print(f"❌ ERROR: No species folders found in {AUDIO_ROOT}")
        return
    
    print(f"Found {len(bird_species)} bird species\n")
    
    # Create class mapping
    class_to_idx = {species.name: idx for idx, species in enumerate(bird_species)}
    idx_to_class = {str(v): k for k, v in class_to_idx.items()}
    
    # Save mappings
    with open(CURRENT_DIR / "class_map.json", "w") as f:
        json.dump(class_to_idx, f, indent=4)
    with open(CURRENT_DIR / "idx_to_class.json", "w") as f:
        json.dump(idx_to_class, f, indent=4)
    
    print(f"Class Mapping:")
    for idx, species in enumerate(bird_species):
        print(f"  {idx}: {species.name}")
    print()
    
    # --- PROCESS ALL AUDIO FILES ---
    data_list = []
    total_files = 0
    successful_files = 0
    
    for species_folder in bird_species:
        species_name = species_folder.name
        species_idx = class_to_idx[species_name]
        
        output_species_dir = PREPROCESSED_OUTPUT / species_name
        output_species_dir.mkdir(parents=True, exist_ok=True)
        
        audio_files = list(species_folder.glob("*.mp3")) + \
                     list(species_folder.glob("*.wav")) + \
                     list(species_folder.glob("*.flac"))
        
        if not audio_files:
            print(f"⚠️  {species_name}: No audio files found")
            continue
        
        print(f"📍 {species_name}: Processing {len(audio_files)} files...")
        
        for audio_file in tqdm(audio_files, desc=f"  {species_name}", leave=False):
            total_files += 1
            
            mel_uint8 = preprocess_audio_to_mel(
                audio_file,
                target_sr=TARGET_SR,
                n_mels=N_MELS,
                n_fft=N_FFT
            )
            
            if mel_uint8 is None:
                continue
            
            output_path = output_species_dir / f"{audio_file.stem}.npy"
            np.save(output_path, mel_uint8)
            
            data_list.append({
                "filepath": str(output_path.resolve()),
                "label": species_idx,
                "bird_name": species_name
            })
            
            successful_files += 1
        
        print(f"  ✓ {len(audio_files)} files → {len([d for d in data_list if d['bird_name'] == species_name])} preprocessed\n")
    
    print(f"\n{'='*70}")
    print(f"PREPROCESSING COMPLETE")
    print(f"  Total files found: {total_files}")
    print(f"  Successfully processed: {successful_files}")
    print(f"  Total samples: {len(data_list)}")
    print(f"{'='*70}\n")
    
    if len(data_list) == 0:
        print("❌ No audio files were successfully processed!")
        return
    
    # --- CREATE DATAFRAME ---
    df = pd.DataFrame(data_list)
    print(f"Original class distribution:")
    print(df['bird_name'].value_counts().sort_index())
    print()
    
    # --- STRATIFIED OVERSAMPLE ---
    print(f"Applying stratified oversampling (min {MIN_SAMPLES_PER_CLASS} per class)...")
    
    oversampled_dfs = []
    for class_label in sorted(df['label'].unique()):
        class_df = df[df['label'] == class_label].reset_index(drop=True)
        class_name = idx_to_class[str(int(class_label))]
        current_count = len(class_df)
        
        if current_count < MIN_SAMPLES_PER_CLASS:
            class_df = resample(
                class_df,
                n_samples=MIN_SAMPLES_PER_CLASS,
                replace=True,
                random_state=SEED
            )
            print(f"  {class_name}: {current_count} → {MIN_SAMPLES_PER_CLASS}")
        else:
            print(f"  {class_name}: {current_count} (no oversample needed)")
        
        oversampled_dfs.append(class_df)
    
    df = pd.concat(oversampled_dfs, ignore_index=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    print(f"\n✅ After oversampling: {len(df)} total samples\n")
    
    # --- SPLIT TRAIN/VAL ---
    train_df, val_df = train_test_split(
        df, test_size=VAL_SIZE, stratify=df['label'], random_state=SEED, shuffle=True
    )
    
    train_df.to_csv(CURRENT_DIR / "train.csv", index=False)
    val_df.to_csv(CURRENT_DIR / "val.csv", index=False)
    
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}\n")
    
    # --- COMPUTE CLASS WEIGHTS ---
    class_counts = train_df['label'].value_counts().sort_index()
    num_classes = len(class_counts)
    class_weights = num_classes / (class_counts.values * len(class_counts))
    
    class_weight_dict = {str(int(idx)): float(weight) 
                         for idx, weight in enumerate(class_weights)}
    
    with open(CURRENT_DIR / "class_weights.json", "w") as f:
        json.dump(class_weight_dict, f, indent=4)
    
    print("Class weights:")
    for idx in sorted([int(k) for k in class_weight_dict.keys()]):
        bird_name = idx_to_class[str(idx)]
        print(f"  {bird_name}: {class_weight_dict[str(idx)]:.4f}")
    
    # --- SAVE PASST STATS ---
    passt_stats = {
        "mean": -4.2677393,
        "std": 4.5689974
    }
    
    with open(CURRENT_DIR / "passt_stats.json", "w") as f:
        json.dump(passt_stats, f, indent=4)
    
    # --- SAVE SUMMARY ---
    summary = {
        "total_samples": len(df),
        "num_classes": num_classes,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "val_split": VAL_SIZE,
        "seed": SEED,
        "min_samples_per_class_after_oversampling": MIN_SAMPLES_PER_CLASS,
        "preprocessed_dir": str(PREPROCESSED_OUTPUT),
        "audio_root_used": str(AUDIO_ROOT),
        "mel_config": {
            "sr": TARGET_SR,
            "n_mels": N_MELS,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH
        },
        "class_distribution": {idx_to_class[str(int(idx))]: int(count) 
                              for idx, count in enumerate(class_counts)},
    }
    
    with open(CURRENT_DIR / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    
    print(f"\n{'='*70}")
    print("✅ ALL DONE!")
    print(f"{'='*70}")
    print(f"\nGenerated files:")
    print(f"  train.csv")
    print(f"  val.csv")
    print(f"  class_map.json")
    print(f"  class_weights.json")
    print(f"  dataset_summary.json")
    print(f"  idx_to_class.json")
    print(f"\nPreprocessed data: {PREPROCESSED_OUTPUT}/")
    print(f"\nNext step: python train.py (after setting PHASE=1)")


if __name__ == "__main__":
    main()
