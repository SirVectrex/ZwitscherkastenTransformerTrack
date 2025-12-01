import numpy as np
import matplotlib.pyplot as plt
import librosa.display

# Pfad zu einer deiner neuen .npy Dateien anpassen
npy_file = "./meine_spektrogramme/990594_qA.npy" 

try:
    # 1. Datei laden
    spec = np.load(npy_file)
    print(f"Shape geladen: {spec.shape}")
    # Erwartet: (128, ca. 1000) für 10 Sekunden

    # 2. Plotten
    plt.figure(figsize=(12, 4))
    
    # Da wir log-mel schon berechnet haben, können wir es direkt anzeigen
    # x_axis='time' rechnet die Spalten automatisch in Sekunden um (basierend auf sr und hop)
    librosa.display.specshow(spec, sr=32000, hop_length=320, x_axis='time', y_axis='mel')
    
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log Mel Spectrogram (PaSST Preprocessing)')
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print("Bitte passe den Pfad 'npy_file' im Skript an eine existierende Datei an.")
except Exception as e:
    print(f"Ein Fehler ist aufgetreten: {e}")