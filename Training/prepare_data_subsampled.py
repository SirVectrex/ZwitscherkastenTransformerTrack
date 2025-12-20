import os
import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
DATA_ROOT = "/dev/shm/schoen/data"    # <-- new structure with subset folders
OUTPUT_DIR = "./Training/"
# ---------------------

def main():
    root_path = Path(DATA_ROOT)

    # --- 1. Detect all subset folders ---
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "class_map.json"), "w") as f:
        json.dump(class_to_idx, f, indent=4)
    print("Saved class_map.json")

    # --- 3. Collect all .npy file paths from all subsets ---
    data_list = []

    print("Scanning subset folders for .npy files...")
    for subset in subset_dirs:
        for cls_name in bird_classes:
            cls_path = subset / cls_name
            if not cls_path.exists():
                continue  # class may not appear in every subset (rare case)

            for npy_file in cls_path.glob("*.npy"):
                data_list.append({
                    "filepath": str(npy_file.resolve()),
                    "label": class_to_idx[cls_name],
                    "bird_name": cls_name
                })

    df = pd.DataFrame(data_list)
    print(f"Total collected .npy files: {len(df)}")

    # --- 4. Train/Validation split ---
    train_df, val_df = train_test_split(
        df,
        test_size=0.1,
        stratify=df['label'],
        random_state=42,
        shuffle=True
    )

    # --- 5. Save CSVs ---
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)

    print(f"Created train.csv ({len(train_df)} samples)")
    print(f"Created val.csv ({len(val_df)} samples)")
    print("DONE!")

if __name__ == "__main__":
    main()
