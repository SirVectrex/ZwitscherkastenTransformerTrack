import os
import pandas as pd
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
# We go 2 levels up (../../) from "Training/SubSampledTrack" to find "DataProcessing"
CURRENT_DIR = Path(__file__).parent.resolve()
DATA_ROOT = CURRENT_DIR / "../../DataProcessing/split_dataset_1000_preprocessed"

# Save CSVs and Json stats in the SAME folder as this script
OUTPUT_DIR = CURRENT_DIR 
VAL_SIZE = 0.1
SEED = 42
# ---------------------

def main():
    print(f"Looking for data in: {DATA_ROOT.resolve()}")
    
    if not DATA_ROOT.exists():
        print(f"ERROR: Data root not found at {DATA_ROOT.resolve()}")
        print("Check your folder structure!")
        return

    # --- 1. Detect Classes ---
    # We scan ALL subfolders (set0, subset_0, etc) to find bird classes
    # We look one level deeper to find the actual bird class folders
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

    # --- 2. Collect Files ---
    data_list = []
    print("Scanning folders for .npy files...")
    
    for subset_dir in subset_folders:
        for cls_name in bird_classes:
            cls_path = subset_dir / cls_name
            if cls_path.exists():
                for npy_file in cls_path.glob("*.npy"):
                    data_list.append({
                        "filepath": str(npy_file.resolve()), # Store ABSOLUTE path
                        "label": class_to_idx[cls_name],
                        "bird_name": cls_name
                    })

    df = pd.DataFrame(data_list)
    print(f"Total samples collected: {len(df)}")

    if len(df) == 0:
        print("CRITICAL: No .npy files found! Check if preprocessing ran correctly.")
        return

    # --- 3. Split ---
    train_df, val_df = train_test_split(
        df, test_size=VAL_SIZE, stratify=df['label'], random_state=SEED, shuffle=True
    )

    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "val.csv", index=False)
    
    # --- 4. Save PaSST Statistics ---
    passt_stats = {
        "mean": -4.2677393,
        "std": 4.5689974
    }
    
    with open(OUTPUT_DIR / "passt_stats.json", "w") as f:
        json.dump(passt_stats, f, indent=4)
        
    print(f"DONE! Files saved to {OUTPUT_DIR}")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")

if __name__ == "__main__":
    main()