# Kepler Exoplanet Detection — KOI Features

NASA's Kepler mission catalogued thousands of potential planets, but many turned out to be false signals from eclipsing binary stars or instrument noise. This project builds a machine learning pipeline to classify Kepler Objects of Interest (KOIs) using physical features — orbital period, planetary radius, equilibrium temperature, stellar properties, and confidence scores — distinguishing confirmed exoplanets from false positives, and ranking unresolved candidates by how likely they are to be real planets.

---

## The Problem

Kepler flagged over 9,000 objects of interest during its mission. Each one had to be reviewed and labeled:

- **Confirmed** — a verified exoplanet
- **False Positive** — a mimicking signal, usually an eclipsing binary star or instrument artifact
- **Candidate** — unresolved, awaiting follow-up confirmation

Manual review is expensive and slow. This project automates the triage using machine learning, and for the ~400 unresolved candidates in the dataset, ranks them by how likely they are to be real planets — helping prioritize which ones deserve follow-up attention first.

---

## Dataset

[Kepler Exoplanet Dataset on Kaggle](https://www.kaggle.com/datasets/gauravkumar2525/kepler-exoplanet-dataset) — sourced from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
  
| Feature | Description |
| :--- | :--- |
| `koi_score` | Confidence score for the planetary classification |
| `koi_period` | Orbital period (days) |
| `koi_prad` | Estimated planetary radius (Earth radii) |
| `koi_teq` | Estimated equilibrium temperature (Kelvin) |
| `koi_insol` | Insolation flux relative to Earth |
| `koi_steff` | Effective temperature of the host star (Kelvin) |
| `koi_srad` | Stellar radius (solar radii) |
| `koi_slogg` | Surface gravity of the host star (log scale) |
| `koi_kepmag` | Brightness of the star as observed by Kepler |

---

## Approach

### Preprocessing
- Iterative imputation for missing values
- Log transformation on skewed features (`koi_period`, `koi_insol`, `koi_prad`, `koi_srad`)
- RobustScaler normalization
- All steps wrapped in a **scikit-learn Pipeline** to prevent data leakage
- SMOTE oversampling to handle class imbalance

### Feature Engineering
- `koi_multiplicity` — number of KOIs sharing the same host star (multi-planet systems are far more likely to be real, known as the Kepler multiplicity boost)
- `score_period` — confidence score weighted by orbital period
- `radius_ratio` — planet to stellar radius ratio
- `score_sq` — squared confidence score
- `temp_ratio` — equilibrium to stellar temperature ratio

### Models
- **Decision Tree** — baseline
- **XGBoost** — tuned via RandomizedSearchCV (40 iterations, 5-fold StratifiedKFold)
- **Stacking Ensemble** — XGBoost + Random Forest with Logistic Regression meta-learner

---

## Results

### Multiclass Classification (FALSE POSITIVE / CANDIDATE / CONFIRMED)

| Model | Accuracy | Macro F1 |
| :---: | :---: | :---: |
| Decision Tree | 76% | 0.71 |
| XGBoost (tuned) | 80% | 0.76 |
| Stacking Ensemble | 80% | 0.76 |

The CANDIDATE class is intentionally the hardest to classify — these are objects that astronomers themselves have not been able to resolve. The model's difficulty with this class reflects the genuine ambiguity of the data, not a modeling failure.

### Binary Classification (FALSE POSITIVE vs CONFIRMED)

Candidates excluded — the binary task focuses on cases where a ground truth label exists.

| Metric | Score |
| :---: | :---: |
| Accuracy | 97% |
| ROC-AUC | 0.995 |
| Average Precision | 0.99 |

### Candidate Priority Ranking

Rather than classifying candidates (which would be scientifically dishonest — astronomers labeled them unresolved for a reason), the model assigns each candidate a probability of being a confirmed planet and ranks them for follow-up priority.

- **42 candidates** ranked high priority (>80% confidence of being a real planet)
- **281 candidates** likely false positives (<20% confidence)
- **13 candidates** genuinely ambiguous (40–60% confidence)

---

## Astrophysical Interpretation

SHAP analysis shows which features matter most to the model, and the results make physical sense:

- **`score_sq`** — the single most important feature, engineered by squaring `koi_score`. Squaring amplifies the gap between high and low confidence objects, making it easier for the model to separate confirmed planets from false positives.
- **`koi_score`** — Kepler's own confidence score, the second strongest signal. The model is essentially agreeing with the telescope's own judgment.
- **`koi_prad`** — planetary radius. Real planets fall within a predictable size range, so unusually large or small values are a red flag.
- **`radius_ratio`** — engineered feature (koi_prad / koi_srad). Comparing the planet's size to its host star's size is more informative than either alone — objects with extreme ratios are more likely eclipsing binary stars mimicking a planet.
- **`koi_multiplicity`** — engineered feature counting how many candidates share the same host star. Stars with multiple candidates are far more likely to host real planets — it's statistically very unlikely for multiple false signals to occur around the same star by coincidence.
- **`koi_period`** — very short orbital periods are more commonly associated with false positives than real planets.

---

## Project Structure

```
/kaggle
    exoplanets_data.csv
/models
    xgb_multiclass.pkl
    xgb_binary.pkl
    stacking_ensemble.pkl
    preprocessing_pipeline.pkl
/notebooks
    kepler_koi_classification.ipynb
/outputs
README.md
requirements.txt
```

---

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/kepler_koi_classification.ipynb
```

---

## Tools & Libraries

Python · XGBoost · scikit-learn · imbalanced-learn · SHAP · pandas · NumPy · Matplotlib · Seaborn
