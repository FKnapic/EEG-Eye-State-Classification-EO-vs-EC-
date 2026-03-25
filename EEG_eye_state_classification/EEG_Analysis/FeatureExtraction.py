import os
import mne
import numpy as np
import pandas as pd
import pywt
from scipy.signal import welch
from scipy.stats import entropy

# Parametri i frekvencijski pojasevi 
BANDS = {
    'delta': (0.5, 3.5),
    'theta': (4, 7),
    'alpha': (8, 13),
    'beta':  (14, 30),
    'gamma': (30, 45)
}

INPUT_DIR = "cleaned_data"
OUTPUT_DIR = "features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#  Funkcija za izračun značajki 
def extract_features(raw, subject_id, eyes_label):
    sfreq = raw.info['sfreq']
    data = raw.get_data(picks='eeg')

    results = []
    for i, ch_name in enumerate(raw.ch_names):
        ch_data = data[i, :]

        # Welch PSD
        freqs, psd = welch(ch_data, sfreq, nperseg=sfreq * 2)

        #  Apsolutna i relativna snaga po pojasima
        total_power = np.trapz(psd[(freqs >= 0.5) & (freqs <= 40)],
                               freqs[(freqs >= 0.5) & (freqs <= 40)]) + 1e-12

        band_features = {}
        for band, (fmin, fmax) in BANDS.items():
            mask = (freqs >= fmin) & (freqs <= fmax)
            abs_power = np.trapz(psd[mask], freqs[mask])
            rel_power = abs_power / total_power
            band_features[band] = (abs_power, rel_power)

        # Valićna entropija (DWT)
        coeffs = pywt.wavedec(ch_data, 'db4', level=6)
        energies = np.array([np.sum(c ** 2) for c in coeffs])
        rel_energies = energies / np.sum(energies)
        wavelet_entropy = -np.sum(rel_energies * np.log(rel_energies + 1e-12))

        # Odabrane značajke 
        alpha_abs = band_features['alpha'][0]
        beta_abs = band_features['beta'][0]
        alpha_rel = band_features['alpha'][1]
        beta_rel = band_features['beta'][1]
        alpha_beta_ratio = alpha_abs / (beta_abs + 1e-12)

        results.append({
            'subject': subject_id,
            'eyes': eyes_label,
            'channel': ch_name,
            'alpha_abs': alpha_abs,
            'beta_abs': beta_abs,
            'alpha_rel': alpha_rel,
            'beta_rel': beta_rel,
            'alpha_beta_ratio': alpha_beta_ratio,
            'wavelet_entropy': wavelet_entropy
        })

    return results


def main():
    all_results = []

    files = sorted(f for f in os.listdir(INPUT_DIR)
                   if f.endswith("_cleaned_raw.fif"))

    for fname in files:
        eyes_label = "EC" if "closed" in fname else "EO"
        subject_id = fname.split("_")[0]
        path = os.path.join(INPUT_DIR, fname)

        print(f"\n>>> Processing {fname} ({eyes_label})")
        print(f"    Reading file: {path}")

        # Sigurno učitavanje (uhvati error)
        try:
            raw = mne.io.read_raw_fif(path, preload=True)
        except Exception as e:
            print(f" !!! ERROR reading {fname}")
            print(f"     Reason: {e}")
            print("     Skipping this file...\n")
            continue

        feats = extract_features(raw, subject_id, eyes_label)
        all_results.extend(feats)

    # Spremanje rezultata
    df = pd.DataFrame(all_results)
    out_path = os.path.join(OUTPUT_DIR, "EEG_features.csv")
    df.to_csv(out_path, index=False)

    print(f"\n✔ Feature extraction COMPLETED.")
    print(f"  Saved to: {out_path}")
    print(df.head())


if __name__ == "__main__":
    main()
