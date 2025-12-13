import os
import shutil
import random

# --- CONFIGURATION ---
SOURCE_DIR = "./mel_spectrograms_uint8"
TARGET_ROOT = "./split_dataset_1000min"   # <-- updated folder name
NUM_FOLDERS = 5
TARGET_SAMPLES_PER_CLASS = 1000
# ---------------------

def downsample_to_fixed_amount(source, target_root, num_folders, target_samples):
    if not os.path.exists(source):
        print(f"ERROR: Source '{source}' not found.")
        return

    classes = [d for d in os.listdir(source) if os.path.isdir(os.path.join(source, d))]
    classes.sort()

    print("Scanning dataset...")

    # Filter classes with at least target_samples
    valid_classes = []
    class_file_counts = {}

    for class_name in classes:
        path = os.path.join(source, class_name)
        files = [f for f in os.listdir(path) if not f.startswith('.')]
        count = len(files)

        class_file_counts[class_name] = count

        if count >= target_samples:
            valid_classes.append(class_name)
        else:
            print(f"Skipping '{class_name}' (only {count}, needs >= {target_samples})")

    if not valid_classes:
        print("ERROR: No class has enough samples!")
        return

    imgs_per_folder = target_samples // num_folders
    if imgs_per_folder == 0:
        print("ERROR: Not enough samples per folder!")
        return

    print("-" * 50)
    print(f"Using classes: {valid_classes}")
    print(f"Downsampling each to exactly {target_samples} samples.")
    print(f"Each subset folder gets {imgs_per_folder} samples per class.")
    print("-" * 50)

    # Process each valid class
    for class_name in valid_classes:
        class_src_path = os.path.join(source, class_name)
        files = [f for f in os.listdir(class_src_path) if not f.startswith('.')]
        
        random.shuffle(files)
        selected_files = files[:target_samples]

        print(f"Class '{class_name}': {class_file_counts[class_name]} -> {len(selected_files)} used")

        # Split into folders
        for i in range(num_folders):
            split_folder_name = f"subset_{i}"
            split_class_path = os.path.join(target_root, split_folder_name, class_name)
            os.makedirs(split_class_path, exist_ok=True)

            start = i * imgs_per_folder
            end = start + imgs_per_folder
            batch = selected_files[start:end]

            for file_name in batch:
                src_file = os.path.join(class_src_path, file_name)
                dst_file = os.path.join(split_class_path, file_name)
                shutil.copy2(src_file, dst_file)

    print("-" * 50)
    print("DONE! Dataset downsampled to fixed number of samples per class.")

if __name__ == "__main__":
    downsample_to_fixed_amount(
        SOURCE_DIR, TARGET_ROOT, NUM_FOLDERS, TARGET_SAMPLES_PER_CLASS
    )
