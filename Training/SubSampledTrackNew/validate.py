"""
Quick sanity check for PaSST model on 200 random .npy samples.

Assumptions:
- You have:
    - val.csv  with columns: filepath,label,bird_name
    - preprocessed_mels/ containing uint8 mel .npy files [128, T]
    - best PaSST weights: checkpoints/best_model_passt_phase1.pth  (adjust below!)
- PaSST model code is available as `build_passt_model(num_classes)`.

Usage:
    python quick_test_passt_random200.py
"""

import random
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


# =========================
# CONFIG
# =========================

CONFIG = {
    "val_csv": "val.csv",
    "num_samples": 200,              # how many random examples to test
    "num_classes": 61,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # adjust this to your real checkpoint filename:
    "checkpoint_path": "checkpoints/best_model_passt_phase1.pth",

    # optional: class index → name mapping file
    "idx_to_class_json": "idx_to_class.json",

    # model builder import path, adjust if needed
    # e.g., from your training script:
    #   from passt_model import build_passt_model
}


# =========================
# MODEL DEFINITION IMPORT
# =========================

def build_passt_model(num_classes: int):
    """
    Placeholder: replace with your actual PaSST builder.
    Example if you used HuggingFace AST/PaSST:

        from transformers import AutoModel
        import torch.nn as nn

        model = AutoModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", trust_remote_code=True)
        model.classifier = nn.Linear(768, num_classes)
        return model

    Make sure this matches exactly what you used for training.
    """
    raise NotImplementedError("Replace build_passt_model() with your real PaSST construction.")


# =========================
# DATA HELPERS
# =========================

def load_random_subset(val_csv: str, num_samples: int) -> pd.DataFrame:
    df = pd.read_csv(val_csv)
    if num_samples > len(df):
        num_samples = len(df)
    # sample without replacement, fixed seed for reproducibility
    df_sample = df.sample(n=num_samples, random_state=42).reset_index(drop=True)
    return df_sample


def load_mel_npy(path: str) -> torch.Tensor:
    """
    Load uint8 mel [128, T] and convert to float32 tensor [1, 128, T] in [0,1].
    """
    mel = np.load(path).astype(np.float32) / 255.0
    mel = torch.from_numpy(mel).unsqueeze(0)   # [1, 128, T]
    return mel


# =========================
# EVALUATION
# =========================

def evaluate_random_samples():
    device = torch.device(CONFIG["device"])
    print(f"Using device: {device}")

    # 1) Load metadata
    val_csv_path = Path(CONFIG["val_csv"])
    if not val_csv_path.exists():
        print(f"❌ val.csv not found at: {val_csv_path}")
        return

    df_sample = load_random_subset(str(val_csv_path), CONFIG["num_samples"])
    print(f"Loaded {len(df_sample)} random validation samples.")

    # optionally load idx→class mapping
    idx_to_class = None
    idx_map_path = Path(CONFIG["idx_to_class_json"])
    if idx_map_path.exists():
        with open(idx_map_path, "r") as f:
            idx_to_class = json.load(f)

    # 2) Build and load model
    print("Building PaSST model...")
    model = build_passt_model(CONFIG["num_classes"]).to(device)
    ckpt_path = Path(CONFIG["checkpoint_path"])
    if not ckpt_path.exists():
        print(f"❌ Checkpoint not found at: {ckpt_path}")
        return

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"✅ Loaded weights from {ckpt_path}")

    # 3) Iterate over random subset
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    examples_to_print = 10
    printed = 0

    with torch.no_grad():
        for _, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="Evaluating"):
            filepath = row["filepath"]
            true_label = int(row["label"])

            mel = load_mel_npy(filepath).to(device)          # [1, 128, T]
            mel = mel.unsqueeze(0)                           # [B=1, 1, 128, T] – adjust if your PaSST expects [B, 1, 128, T]

            logits = model(mel)                              # [1, num_classes]
            probs = F.softmax(logits, dim=1)
            top1 = torch.argmax(probs, dim=1).item()
            top5_vals, top5_idx = torch.topk(probs, k=5, dim=1)

            total += 1
            if top1 == true_label:
                correct_top1 += 1
            if true_label in top5_idx[0].tolist():
                correct_top5 += 1

            # print a few example predictions
            if printed < examples_to_print:
                printed += 1
                true_name = idx_to_class.get(str(true_label), str(true_label)) if idx_to_class else str(true_label)
                pred_name = idx_to_class.get(str(top1), str(top1)) if idx_to_class else str(top1)
                print(f"\nSample #{printed}")
                print(f"  File:       {filepath}")
                print(f"  True label: {true_label} ({true_name})")
                print(f"  Pred label: {top1} ({pred_name})")
                print(f"  Top-5 idx:  {top5_idx[0].tolist()}")

    acc1 = 100.0 * correct_top1 / total
    acc5 = 100.0 * correct_top5 / total

    print("\n" + "=" * 60)
    print(f"Random {total}‑sample check on val set")
    print(f"Top‑1 accuracy: {acc1:.2f} %")
    print(f"Top‑5 accuracy: {acc5:.2f} %")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_random_samples()
