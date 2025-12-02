import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# KONFIGURATION
# ==========================================
API_KEY = "93bfa6c3854af3b6f7429d1b36e5da0ac5491032"

# Wir suchen nach einer Kohlmeise (Parus major), da diese garantiert existiert
SPECIES_QUERY = "gen:Parus sp:major q:A len:10-40"
API_BASE = "https://xeno-canto.org/api/2/recordings"

def get_representative_file_url():
    """
    Holt eine ECHTE Datei aus der API, diesmal MIT KEY.
    """
    print("🔍 Suche eine repräsentative Datei (Parus major)... ", end="", flush=True)
    try:
        # Hier übergeben wir jetzt den KEY!
        params = {
            "query": SPECIES_QUERY,
            "key": API_KEY
        }
        r = requests.get(API_BASE, params=params, timeout=10)
        
        if r.status_code != 200:
            print(f"Fehler: API Status {r.status_code}")
            return None
            
        data = r.json()
        
        # Sicherheits-Check: Ist der Key überhaupt da?
        if 'numRecordings' not in data:
            print("\n❌ API Antwort unerwartet (kein 'numRecordings').")
            print(f"Antwort-Ausschnitt: {str(data)[:100]}")
            return None

        if int(data['numRecordings']) == 0:
            print("Keine Aufnahmen gefunden.")
            return None
            
        # Wir nehmen die erste Datei -> Das ist unser Testobjekt
        test_url = data['recordings'][0]['file']
        print("✅ Gefunden!")
        return test_url
        
    except Exception as e:
        print(f"❌ Fehler bei der Suche: {e}")
        return None

def benchmark_search(workers, num_requests=8):
    """Testet die API-Suchgeschwindigkeit MIT KEY"""
    start = time.time()
    successful = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Auch hier den Key mitschicken
        params = {"query": SPECIES_QUERY, "key": API_KEY}
        futures = [executor.submit(requests.get, API_BASE, params=params, timeout=10) for _ in range(num_requests)]
        for f in as_completed(futures):
            try:
                if f.result().status_code == 200:
                    successful += 1
            except: pass
    return time.time() - start, successful

def benchmark_download(url, workers, num_files=16):
    """Testet den Download-Durchsatz"""
    start = time.time()
    total_bytes = 0
    successful = 0
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(requests.get, url, timeout=30) for _ in range(num_files)]
        for f in as_completed(futures):
            try:
                resp = f.result()
                if resp.status_code == 200:
                    total_bytes += len(resp.content)
                    successful += 1
            except: pass
            
    duration = time.time() - start
    if duration < 0.001: duration = 0.001
        
    mb_per_sec = (total_bytes / (1024*1024)) / duration
    return mb_per_sec, successful

# ==========================================
# HAUPTPROGRAMM
# ==========================================
if __name__ == "__main__":
    print("========================================")
    print(" 🚀 REALISTISCHER PERFORMANCE CHECK")
    print("========================================")

    # 1. Echte URL holen
    real_url = get_representative_file_url()
    
    if not real_url:
        print("\nKritischer Fehler: Konnte keine Test-Datei finden.")
        print("Das Netzwerk blockt uns evtl. trotz API-Key.")
        exit()

    # --- TEST 1: SUCHE ---
    print("\n--- TEST 1: Optimale SEARCH_WORKERS ---")
    search_configs = [1, 2, 4, 8]
    best_search_workers = 1
    best_search_time = float('inf')

    for w in search_configs:
        print(f"Teste {w} Worker... ", end="", flush=True)
        dur, success = benchmark_search(w, num_requests=8)
        print(f"Dauer: {dur:.2f}s ({success}/8 ok)")
        
        if success == 8 and dur < best_search_time:
            best_search_time = dur
            best_search_workers = w

    print(f"✅ Empfehlung: SEARCH_WORKERS = {best_search_workers}")


    # --- TEST 2: DOWNLOAD ---
    print("\n--- TEST 2: Optimale DOWNLOAD_WORKERS ---")
    download_configs = [4, 8, 16, 24, 32, 64]
    best_dl_workers = 4
    max_throughput = 0

    for w in download_configs:
        print(f"Teste {w} Worker... ", end="", flush=True)
        speed, success = benchmark_download(real_url, w, num_files=16)
        print(f"Speed: {speed:.2f} MB/s ({success}/16 ok)")
        
        if success < 12:
            print("  -> Zu viele Timeouts. Das Netzwerk ist überlastet.")
            break
            
        if speed > max_throughput:
            max_throughput = speed
            best_dl_workers = w
        else:
            if speed < max_throughput * 1.05:
                print(f"  -> Sättigung erreicht.")
                break

    print(f"✅ Empfehlung: DOWNLOAD_WORKERS = {best_dl_workers}")

    print("\n========================================")
    print(" 🏁 ERGEBNIS FÜR DEIN SKRIPT:")
    print(f" SEARCH_WORKERS   = {best_search_workers}")
    print(f" DOWNLOAD_WORKERS = {best_dl_workers}")
    print("========================================")