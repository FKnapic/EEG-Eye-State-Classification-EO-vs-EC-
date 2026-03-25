import mne
import os

#  putanja do filtriranih podataka
input_dir = "filtered_data"
output_dir = "cleaned_data"
os.makedirs(output_dir, exist_ok=True)

# iteracija kroz sve filtrirane datoteke
for file in os.listdir(input_dir):
    if not file.endswith("_filtered_raw.fif"):
        continue

    subj = file.split("_")[0]
    condition = "eyesopen" if "eyesopen" in file else "eyesclosed"
    print(f"\nProcessing ICA for: {subj} ({condition})")

    # filtrirani EEG
    raw = mne.io.read_raw_fif(os.path.join(input_dir, file), preload=True)

    #  odaberi EEG kanale
    raw.pick_types(eeg=True, eog=True)

    # inicijalizacija ICA
    ica = mne.preprocessing.ICA(n_components=60, method='fastica', random_state=97)
    ica.fit(raw)

    #  pronađi EOG artefakte
    eog_inds, scores = ica.find_bads_eog(raw, ch_name=['Fp1', 'Fp2'])
    print(f"  Found {len(eog_inds)} EOG components: {eog_inds}")

    #  ukloni te komponente
    ica.exclude = eog_inds
    raw_clean = ica.apply(raw)

    #  Spremi očišćeni EEG
    save_path = os.path.join(output_dir, file.replace("_filtered_raw", "_cleaned_raw"))
    raw_clean.save(save_path, overwrite=True)
    print(f"  Saved cleaned EEG: {save_path}")

print("\n ICA processing completed for all subjects!")
