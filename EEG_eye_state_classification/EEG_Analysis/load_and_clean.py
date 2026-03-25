import mne
import os
import subprocess
root_dir = "data"

output_dir = "filtered_data"
os.makedirs(output_dir, exist_ok=True)

#iteracija kroz ispitanike
for subj in os.listdir(root_dir):
    subj_path = os.path.join(root_dir, subj, "ses-session1", "eeg")
    if not os.path.isdir(subj_path):
        continue

    print(f"\n Processing subject: {subj}")

    # Dva tipa zadataka: eyes open i eyes closed
    for condition in ["eyesopen", "eyesclosed"]:
        vhdr_file = f"{subj}_ses-session1_task-{condition}_eeg.vhdr"
        vhdr_path = os.path.join(subj_path, vhdr_file)

        if not os.path.exists(vhdr_path):
            print(f" Skipping {condition} — file not found for {subj}")
            continue

        print(f"  ➜ Loading {condition} data...")

        # Učitaj sirovi EEG
        raw = mne.io.read_raw_brainvision(vhdr_path, preload=True)

        # postavljanje norme
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore')

        # Rereferenciraj na prosjek svih kanala
        raw.set_eeg_reference('average')

        # filtriranje između 1 - 40 hz
        print("  ➜ Filtering 1–40 Hz...")
        raw.filter(1., 40., fir_design='firwin')

        # spremanje filtriranog eega
        save_path = os.path.join(output_dir, f"{subj}_{condition}_filtered_raw.fif")
        raw.save(save_path, overwrite=True)
        print(f"  Saved: {save_path}")

print("\n All subjects filtered successfully!")

#pokretanje ICA 
subprocess.run(["python", "EEG_Analysis/run_ica.py"])

