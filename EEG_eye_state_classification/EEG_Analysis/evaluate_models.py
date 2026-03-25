import os
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

MODELS_DIR = "models"
DATA_DIR = "data"


def evaluate_model(model_name):
    print("\n======================================")
    print(f" Evaluating model: {model_name.upper()}")
    print("======================================")

    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"), allow_pickle=True)
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"), allow_pickle=True)   

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))


def main():
    for model_name in ["logreg", "svm", "rf"]:
        evaluate_model(model_name)


if __name__ == "__main__":
    main()
