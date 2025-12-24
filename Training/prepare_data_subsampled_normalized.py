import os
import pandas as pd
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_ROOT = "/dev/shm/schoen/data"      # Input data (RAM)
OUTPUT_DIR = "/dev/shm/schoen/output"   # Output CSVs + Stats (RAM)
STATS_SAMPLE_SIZE = 5000                # (Wird hier nicht mehr gebraucht, stört aber nicht)
# ---------------------

def compute_statistics(file_list):
    """
    (Veraltet/Optional) Berechnet Mean/Std selbst.
    Wird unten nicht mehr aufgerufen, wenn wir AudioSet-Werte nutzen.
    """
    print(f"Computing Mean & Std based on a sample of {len(file_list)} files...")
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    total_pixels = 0
    
    for filepath in tqdm(file_list, desc="Calculating Stats"):
        try:
            mel = np.load(filepath)
            pixel_sum += np.sum(mel)
            pixel_sq_sum += np.sum(mel ** 2)
            total_pixels += mel.size
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue

    if total_pixels == 0:
        return 0.0, 1.0

    global_mean = pixel_sum / total_pixels
    global_var = (pixel_sq_sum / total_pixels) - (global_mean ** 2)
    global_std = np.sqrt(global_var)

    return float(global_mean), float(global_std)


def main():
    root_path = Path(DATA_ROOT)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1. Detect all subset folders ---
    print(f"Scanning {DATA_ROOT}...")
    subset_dirs = sorted(
        [d for d in root_path.iterdir() if d.is_dir() and d.name.startswith("subset_")]
    )
    if not subset_dirs:
        print(f"ERROR: No subset folders found in {DATA_ROOT}")
        return

    print(f"Found {len(subset_dirs)} subset folders:", [d.name for d in subset_dirs])

    # --- 2. Detect all bird classes inside subsets ---
    bird_classes = set()
    for subset in subset_dirs:
        for cls_dir in subset.iterdir():
            if cls_dir.is_dir():
                bird_classes.add(cls_dir.name)

    bird_classes = sorted(list(bird_classes))

    if not bird_classes:
        print("ERROR: No bird class folders found inside subsets.")
        return

    print(f"Detected {len(bird_classes)} bird species.")

    # Create class → index mapping
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(bird_classes)}

    # Save mapping
    with open(os.path.join(OUTPUT_DIR, "class_map.json"), "w") as f:
        json.dump(class_to_idx, f, indent=4)
    print("Saved class_map.json")

    # --- 3. Collect all .npy file paths ---
    data_list = []
    print("Scanning subset folders for .npy files...")
    for subset in subset_dirs:
        for cls_name in bird_classes:
            cls_path = subset / cls_name
            if not cls_path.exists():
                continue

            for npy_file in cls_path.glob("*.npy"):
                data_list.append({
                    "filepath": str(npy_file.resolve()),
                    "label": class_to_idx[cls_name],
                    "bird_name": cls_name
                })

    df = pd.DataFrame(data_list)
    print(f"Total collected .npy files: {len(df)}")
    
    if len(df) == 0:
        print("ABORTING: No .npy files found.")
        return

    # --- 4. Train/Validation split ---
    train_df, val_df = train_test_split(
        df,
        test_size=0.1,
        stratify=df['label'],
        random_state=42,
        shuffle=True
    )

    # --- 5. STATS SETUP (GEÄNDERT) ---
    print("-" * 30)
    print("Using OFFICIAL AudioSet Statistics (PaSST Defaults)")
    
    # Anstatt zu berechnen, setzen wir die Werte hart:
    mean = -4.2677393
    std = 4.5689974
    
    # HINWEIS: Falls du doch wieder selbst berechnen willst, 
    # kommentiere die 2 Zeilen oben aus und nimm das hier wieder rein:
    # mean, std = compute_statistics(train_df['filepath'].sample(min(len(train_df), STATS_SAMPLE_SIZE)).tolist())

    print(f"Set Dataset Mean: {mean:.4f}")
    print(f"Set Dataset Std:  {std:.4f}")
    print("-" * 30)

    # Save stats to JSON so train.py can read them
    stats = {"mean": mean, "std": std}
    with open(os.path.join(OUTPUT_DIR, "normalization_stats.json"), "w") as f:
        json.dump(stats, f, indent=4)
    print("Saved normalization_stats.json with AudioSet values.")

    # --- 6. Save CSVs ---
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)

    print(f"Created train.csv ({len(train_df)} samples)")
    print(f"Created val.csv ({len(val_df)} samples)")
    print("DONE!")

if __name__ == "__main__":
    main()
