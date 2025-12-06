import torch
import sys

print(f"Python version: {sys.version.split()[0]}")
print(f"PyTorch version: {torch.__version__}")
print("-" * 30)

if torch.cuda.is_available():
    print("✅ CUDA is available!")
    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"Current Device ID: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("❌ CUDA is NOT available.")
    print("You are currently running on CPU.")
    print("If you have an NVIDIA GPU, make sure you installed the CUDA version of PyTorch.")
    print("Command to fix (usually): pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")