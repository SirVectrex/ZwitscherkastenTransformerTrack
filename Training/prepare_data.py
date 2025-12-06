import os
import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
# Point this to your folder containing the bird subfolders
DATA_ROOT = "./DataProcessing/mel_spectrograms_uint8" 
OUTPUT_DIR = "./Training/" 

def main():
    root_path = Path(DATA_ROOT)
    
    # 1. Scan the directory to find classes (birdtypes)
    # Sort them to ensure label '0' is always the same bird every time you run this.
    classes = sorted([d.name for d in root_path.iterdir() if d.is_dir()])
    
    if not classes:
        print(f"Error: No folders found in {DATA_ROOT}")
        return

    print(f"Found {len(classes)} bird types.")
    
    # Create a mapping: BirdName -> Integer
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    
    # Save this mapping so you can interpret predictions later!
    with open(os.path.join(OUTPUT_DIR, "class_map.json"), "w") as f:
        json.dump(class_to_idx, f, indent=4)
    print("Saved class_map.json")

    # 2. Collect all file paths
    data_list = []
    
    print("Scanning files...")
    for cls_name in classes:
        cls_folder = root_path / cls_name
        class_idx = class_to_idx[cls_name]
        
        # Find all .npy files
        for npy_file in cls_folder.glob("*.npy"):
            data_list.append({
                "filepath": str(npy_file.resolve()),
                "label": class_idx,
                "bird_name": cls_name
            })
            
    df = pd.DataFrame(data_list)
    print(f"Total files found: {len(df)}")

    # 3. Split into Train and Validation
    # stratify=df['label'] ensures that if you have a rare bird, 
    # it appears in both train and val sets proportionally.
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2,      # 20% for validation
        stratify=df['label'], 
        random_state=42,
        shuffle=True
    )

    # 4. Save to CSV
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
    
    print(f"Created train.csv ({len(train_df)} samples) and val.csv ({len(val_df)} samples).")

if __name__ == "__main__":
    main()