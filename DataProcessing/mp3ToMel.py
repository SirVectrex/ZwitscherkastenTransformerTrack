import librosa
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm  # Für den Fortschrittsbalken 
import concurrent.futures # Für parallele Verarbeitung

def get_passt_spectrogram(
    file_path, 
    target_sr=32000, 
    target_duration=10.0, 
    n_mels=128
):
    """
    Lädt MP3, wendet Repeat-Padding an und berechnet Log Mel Spektrogramm.
    """
    try:
        # 1. Laden & Resampling (32kHz Mono) [cite: 142]
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    except Exception as e:
        # Korrupte Dateien abfangen
        return None

    if len(y) == 0:
        return None

    # Ziel-Länge in Samples (32000 * 10 = 320.000)
    target_samples = int(target_sr * target_duration)
    current_samples = len(y)

    # 2. Längen-Anpassung (Repeat Padding)
    if current_samples < target_samples:
        n_repeats = int(np.ceil(target_samples / current_samples))
        y_padded = np.tile(y, n_repeats)
        y = y_padded[:target_samples]
    elif current_samples > target_samples:
        y = y[:target_samples]
    
    # 3. Mel Parameter [cite: 143]
    # 25ms Window, 10ms Hop
    n_fft = int(0.025 * target_sr)     # 800 samples
    hop_length = int(0.010 * target_sr) # 320 samples

    # 4. Spektrogramm Berechnung
    mel_spec = librosa.feature.melspectrogram(
        y=y, 
        sr=sr, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        n_mels=n_mels
    )

    # 5. Logarithmierung (dB)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    return log_mel_spec

def process_file(file_info):
    """Hilfsfunktion für die Parallelisierung"""
    src_path, dst_path = file_info
    
    # Wenn Zieldatei schon existiert, überspringen (Resume-Funktion)
    if dst_path.exists():
        return "Skipped"

    spec = get_passt_spectrogram(src_path)
    
    if spec is not None:
        # Ordner erstellen, falls nicht existent
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        # Als .npy speichern
        np.save(dst_path, spec)
        return "Success"
    else:
        return "Error"

# --- HAUPTPROGRAMM ---

if __name__ == "__main__":
    # Konfiguration
    INPUT_ROOT = Path("./audio_data")       # Wo liegen die MP3s?
    OUTPUT_ROOT = Path("./mel_spectrograms") # Wo sollen die NPYs hin?
    NUM_WORKERS = 8  # Anzahl CPU-Kerne für Parallelisierung (Audio ist rechenintensiv!)

    print(f"Suche MP3-Dateien in {INPUT_ROOT}...")
    
    # Alle MP3s rekursiv finden (auch in Unterordnern)
    all_mp3s = list(INPUT_ROOT.rglob("*.mp3"))
    
    if not all_mp3s:
        print("Keine MP3-Dateien gefunden!")
        exit()

    print(f"{len(all_mp3s)} Dateien gefunden. Starte Verarbeitung...")

    # Job-Liste erstellen
    # Wir berechnen den Zielpfad VORHER, damit wir die Struktur spiegeln können
    jobs = []
    for mp3_path in all_mp3s:
        # Relativen Pfad ermitteln (z.B. "Parus_major/123.mp3")
        rel_path = mp3_path.relative_to(INPUT_ROOT)
        
        # Zielpfad bauen (z.B. "mel_spectrograms/Parus_major/123.npy")
        npy_path = OUTPUT_ROOT / rel_path.with_suffix('.npy')
        
        jobs.append((mp3_path, npy_path))

    # Parallele Verarbeitung starten
    # ProcessPoolExecutor ist hier wichtig, da Librosa CPU-Lastig ist (Multiprocessing)
    # ThreadPool hilft hier wenig wegen Python GIL
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(tqdm(executor.map(process_file, jobs), total=len(jobs), unit="file"))

    print("\nFertig!")