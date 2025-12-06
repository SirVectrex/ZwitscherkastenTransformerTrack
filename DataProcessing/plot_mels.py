import sys
import numpy as np

import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg" if you have PyQt5 installed

import matplotlib.pyplot as plt

# Usage:
#   python plot_mels.py file1.npy
#   python plot_mels.py file1.npy file2.npy
#   python plot_mels.py file1.npy file2.npy file3.npy

def main(paths):
    n = len(paths)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, path in zip(axes, paths):
        mel = np.load(path).astype(np.float32)
        # uint8 [0, 255] -> dB in [-80, 0]
        mel_db = (mel / 255.0) * 80.0 - 80.0
        im = ax.imshow(mel_db, origin="lower", aspect="auto", cmap="magma")
        ax.set_title(path)
        fig.colorbar(im, ax=ax, format="%+2.1f dB")

    axes[-1].set_xlabel("Frames")
    axes[0].set_ylabel("Mel bins")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("Usage: python plot_mels.py file1.npy [file2.npy] [file3.npy]")
        sys.exit(1)
    main(sys.argv[1:])
