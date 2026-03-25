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

## Results

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

##  Dataset

The dataset is available at:
https://github.com/OpenNeuroDatasets/ds004148

This dataset uses git-annex, meaning large EEG files are not stored directly in the repository and must be downloaded separately.

## Project Structure

project-root/
│
├── data/                # dataset (created during setup)
├── setup/
│   ├── PreuzimanjeSaPoveznica.txt
│   └── Skripta.txt
└── README.md

## Setup

1. Clone the dataset

 cd <project-path>
 git clone https://github.com/OpenNeuroDatasets/ds004148.git data
 cd data


2. Initialize git-annex
 git annex init "local-machine"

Run the download script (PreuzimanjeSaPoveznica.txt)
Go back to the project root:

 cd ..

Run the script:
python setup/Skripta.txt

#Result
fter completion, the data will be available in: data/sub-XX/ses-session1/eeg/
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

