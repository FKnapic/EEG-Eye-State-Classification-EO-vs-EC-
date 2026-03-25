##  Overview

Electroencephalography (EEG) signals contain valuable information about brain activity but are often corrupted by noise and artifacts.
This project applies signal processing techniques and machine learning models to classify:

* **EO (Eyes Open)**
* **EC (Eyes Closed)**

---

##  Pipeline

The full pipeline consists of the following steps:

```
Raw EEG → Filtering → ICA → Feature Extraction → ML Models → Evaluation
```

### 1. Preprocessing

* Standard 10–20 electrode montage applied
* Re-referencing to average reference
* Bandpass filtering (1–40 Hz)
* Artifact removal using **ICA (FastICA)**

### 2. Feature Extraction

Features are extracted in the frequency domain:

* **Absolute power** (alpha, beta)
* **Relative power**
* **Alpha/Beta ratio**
* **Wavelet entropy**

These features are computed for each EEG channel.

### 3. Machine Learning

Three models were used:

* Logistic Regression

* Support Vector Machine (SVM)

* Random Forest

* Data split: **80% training / 20% testing**

* Cross-validation: **5-fold**

* Hyperparameter tuning: **RandomizedSearchCV**

---

## 📊 Results

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 76.64%   |
| SVM                 | 78.55%   |
| Random Forest       | 79.92%   |

* **Best model:** Random Forest
* Balanced dataset: 732 EO / 732 EC samples
* Non-linear models outperform linear models

---

## Key Insights

* EEG signals are inherently **non-linear**, which explains better performance of SVM and Random Forest
* Alpha and beta bands play a crucial role in distinguishing eye states:

  * **Alpha** → dominant in relaxed/closed-eye states
  * **Beta** → dominant in active/open-eye states
* Wavelet entropy captures signal complexity and improves classification

---

## 🚀 How to Run

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Run full pipeline

```
python src/main.py
```

---

##  Dataset


You can download it here:
https://openneuro.org/datasets/ds004148/versions/1.0.1


##  Technologies Used

* Python
* MNE (EEG processing)
* NumPy / Pandas
* SciPy
* PyWavelets
* Scikit-learn

---

## Evaluation Metrics

* Accuracy
* Precision / Recall / F1-score
* Confusion matrix

---

## Conclusion

The results demonstrate that combining EEG spectral features with classical machine learning models enables effective classification of eye states.

The Random Forest model achieved the best performance (~80% accuracy), confirming the importance of non-linear approaches in EEG signal analysis.

---

