import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

FEATURES_PATH = "features/EEG_features.csv"
MODELS_DIR = "models"
DATA_DIR = "data"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# UČITAVANJE PODATAKA
def load_data():
    df = pd.read_csv(FEATURES_PATH)

    feature_cols = [
        "alpha_abs",
        "beta_abs",
        "alpha_rel",
        "beta_rel",
        "alpha_beta_ratio",
        "wavelet_entropy"
    ]

    X = df[feature_cols].values
    y = df["eyes"].values
    return X, y


# TRENING + OPTIMIZACIJA

def train_all_models():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # spremanje TEST skup 
    np.save(os.path.join(DATA_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(DATA_DIR, "y_test.npy"), y_test)

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples:  {len(X_test)}\n")

    models = {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000))
        ]),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True))
        ]),
        "rf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=42))
        ])
    }

    param_grids = {
        "logreg": {
            "clf__C": [0.01, 0.1, 1, 10, 100]
        },
        "svm": {
            "clf__C": [0.1, 1, 10, 100],
            "clf__gamma": ["scale", 0.01, 0.1]
        },
        "rf": {
            "clf__n_estimators": [100, 300, 500],
            "clf__max_depth": [None, 5, 10, 20]
        }
    }

    for name, model in models.items():
        print("=" * 60)
        print(f"Training model: {name.upper()}")
        print("=" * 60)

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grids[name],
            n_iter=15,
            cv=5,
            scoring="accuracy",
            random_state=42,
            n_jobs=-1
        )

        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        y_pred = best_model.predict(X_test)

        print(f"Best CV accuracy: {search.best_score_:.4f}")
        print(f"Test accuracy:    {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification report:")
        print(classification_report(y_test, y_pred))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Best parameters:")
        print(search.best_params_)

        model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
        joblib.dump(best_model, model_path)
        print(f"\nModel saved to: {model_path}\n")

    print("✔ Training completed successfully!")


if __name__ == "__main__":
    train_all_models()
