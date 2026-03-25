import subprocess
import sys
import os
import shutil
import joblib
import numpy as np
from sklearn.metrics import accuracy_score

MODELS_DIR = "models"
DATA_DIR = "data"
ANALYSIS_DIR = "EEG_Analysis"


# Run external Python script
def run_script(script_name):
    path = os.path.join(ANALYSIS_DIR, script_name)
    print("\n" + "=" * 60)
    print(f"▶ Running: {path}")
    print("=" * 60)
    subprocess.run([sys.executable, path], check=True)


# Clean old outputs (safe, reproducible run)

def clean_outputs():
    print("\n" + "=" * 60)
    print(" CLEANING PREVIOUS OUTPUTS")
    print("=" * 60)

    paths_to_remove = [
        "filtered_data",
        "cleaned_data",
        "features",
        "models",
        os.path.join("data", "X_test.npy"),
        os.path.join("data", "y_test.npy"),
    ]

    for path in paths_to_remove:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
                print(f"Removed directory: {path}")
            else:
                os.remove(path)
                print(f"Removed file:      {path}")
        else:
            print(f"Not found (skip):   {path}")



# Evaluacija
    print("\n" + "=" * 60)
    print(" FINAL MODEL ACCURACY SUMMARY")
    print("=" * 60)

    X_test_path = os.path.join(DATA_DIR, "X_test.npy")
    y_test_path = os.path.join(DATA_DIR, "y_test.npy")

    if not (os.path.exists(X_test_path) and os.path.exists(y_test_path)):
        print(" Test set not found. Training may have failed.")
        return

    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path, allow_pickle=True)

    print("\nAccuracy (EO vs EC):")
    print("-" * 40)

    for model_name in ["logreg", "svm", "rf"]:
        model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")

        if not os.path.exists(model_path):
            print(f"{model_name.upper():10s}: (missing model)")
            continue

        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{model_name.upper():10s}: {acc:.4f}")

    print("-" * 40)



def main():
    clean_outputs()

    # 1. Load + filter EEG + ICA
    run_script("load_and_clean.py")

    # 2. Feature extraction
    run_script("FeatureExtraction.py")

    # 3. Training + hyperparameter optimization
    run_script("train_models.py")
   
    # 4. Final evaluation summary
    run_script("evaluate_models.py")
    
    print("\n✔ COMPLETE EEG PIPELINE FINISHED SUCCESSFULLY ")


if __name__ == "__main__":
    main()
