import os
import pandas as pd
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from collections import Counter


# --- CONFIGURATION ---
CURRENT_DIR = Path(__file__).parent.resolve()
DATA_ROOT = CURRENT_DIR / "../../DataProcessing/split_dataset_1000_preprocessed"

OUTPUT_DIR = CURRENT_DIR 
VAL_SIZE = 0.1
SEED = 42

# Oversampling config: bring minority classes up to this threshold
MIN_SAMPLES_PER_CLASS = 500
# ---------------------


def stratified_oversample(df, idx_to_class, min_samples_per_class=500):
    """
    Oversample rare classes to at least min_samples_per_class.
    Common classes stay as-is to avoid introducing redundancy.
    """
    print(f"\n📊 Original class distribution:")
    original_counts = df['label'].value_counts().sort_index()
    print(original_counts)
    
    oversampled_dfs = []
    
    for class_label in sorted(df['label'].unique()):
        class_df = df[df['label'] == class_label].reset_index(drop=True)
        class_name = idx_to_class[str(int(class_label))]
        current_count = len(class_df)
        
        if current_count < min_samples_per_class:
            # Oversample with replacement
            class_df = resample(
                class_df, 
                n_samples=min_samples_per_class, 
                replace=True,
                random_state=SEED
            )
            print(f"  {class_name}: {current_count} → {min_samples_per_class} samples")
        else:
            print(f"  {class_name}: {current_count} samples (no oversampling needed)")
        
        oversampled_dfs.append(class_df)
    
    df_oversampled = pd.concat(oversampled_dfs, ignore_index=True)
    df_oversampled = df_oversampled.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    print(f"\n✅ After oversampling: {len(df_oversampled)} total samples")
    print(f"Original: {len(df)} samples\n")
    
    return df_oversampled


def main():
    print(f"Looking for data in: {DATA_ROOT.resolve()}")
    
    if not DATA_ROOT.exists():
        print(f"ERROR: Data root not found at {DATA_ROOT.resolve()}")
        print("Check your folder structure!")
        return

    # --- 1. DETECT CLASSES ---
    subset_folders = [d for d in DATA_ROOT.iterdir() if d.is_dir()]
    
    if not subset_folders:
        print("ERROR: No subset/set folders found in data root.")
        return

    print(f"Found {len(subset_folders)} data subsets: {[d.name for d in subset_folders]}")

    # Collect all unique class names from the first valid subset
    first_subset = subset_folders[0]
    bird_classes = sorted([d.name for d in first_subset.iterdir() if d.is_dir()])
    
    if not bird_classes:
        print("ERROR: No bird class folders found inside the subsets.")
        return

    print(f"Found {len(bird_classes)} bird classes.")
    
    # Map class names to indices
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(bird_classes)}
    
    # Save class map locally
    with open(OUTPUT_DIR / "class_map.json", "w") as f:
        json.dump(class_to_idx, f, indent=4)
    
    # Reverse mapping for later reference
    idx_to_class = {str(v): k for k, v in class_to_idx.items()}
    with open(OUTPUT_DIR / "idx_to_class.json", "w") as f:
        json.dump(idx_to_class, f, indent=4)

    # --- 2. COLLECT FILES ---
    data_list = []
    print("Scanning folders for .npy files...")
    
    for subset_dir in subset_folders:
        for cls_name in bird_classes:
            cls_path = subset_dir / cls_name
            if cls_path.exists():
                for npy_file in cls_path.glob("*.npy"):
                    data_list.append({
                        "filepath": str(npy_file.resolve()),
                        "label": class_to_idx[cls_name],
                        "bird_name": cls_name
                    })

    df = pd.DataFrame(data_list)
    print(f"Total samples collected: {len(df)}")

    if len(df) == 0:
        print("CRITICAL: No .npy files found! Check if preprocessing ran correctly.")
        return

    # --- 3. OVERSAMPLE MINORITY CLASSES ---
    df = stratified_oversample(df, idx_to_class, min_samples_per_class=MIN_SAMPLES_PER_CLASS)

    # --- 4. SPLIT INTO TRAIN/VAL ---
    train_df, val_df = train_test_split(
        df, test_size=VAL_SIZE, stratify=df['label'], random_state=SEED, shuffle=True
    )

    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "val.csv", index=False)
    
    # --- 5. CALCULATE AND SAVE CLASS WEIGHTS ---
    # For weighted loss: weight = total_samples / (num_classes * class_samples)
    class_counts = train_df['label'].value_counts().sort_index()
    num_classes = len(class_counts)
    class_weights = num_classes / (class_counts.values * len(class_counts))
    
    class_weight_dict = {str(int(idx)): float(weight) for idx, weight in enumerate(class_weights)}
    
    with open(OUTPUT_DIR / "class_weights.json", "w") as f:
        json.dump(class_weight_dict, f, indent=4)
    
    print("\n📈 Class weights (for loss function):")
    for idx in sorted([int(k) for k in class_weight_dict.keys()]):
        bird_name = idx_to_class[str(idx)]
        print(f"  {bird_name}: {class_weight_dict[str(idx)]:.4f}")

    # --- 6. SAVE PASST STATISTICS ---
    passt_stats = {
        "mean": -4.2677393,
        "std": 4.5689974
    }
    
    with open(OUTPUT_DIR / "passt_stats.json", "w") as f:
        json.dump(passt_stats, f, indent=4)
    
    # --- 7. SAVE SUMMARY ---
    summary = {
        "total_samples": len(df),
        "num_classes": num_classes,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "val_split": VAL_SIZE,
        "seed": SEED,
        "min_samples_per_class_after_oversampling": MIN_SAMPLES_PER_CLASS,
        "class_distribution": {idx_to_class[str(int(idx))]: int(count) 
                              for idx, count in enumerate(class_counts)},
    }
    
    with open(OUTPUT_DIR / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"\n✅ DONE! Files saved to {OUTPUT_DIR}")
    print(f"   Train samples: {len(train_df)}")
    print(f"   Val samples: {len(val_df)}")
    print(f"   Classes: {num_classes}")


if __name__ == "__main__":
    main()