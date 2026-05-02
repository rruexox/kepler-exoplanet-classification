# Kepler Exoplanet Detection — KOI Features

A machine learning pipeline to classify exoplanet candidates from NASA's Kepler Space Telescope into three categories: FALSE POSITIVE, CANDIDATE, and CONFIRMED — and rank unresolved candidates by their likelihood of being real planets.

---

## Overview

NASA's Kepler Space Telescope identified thousands of potential planets by monitoring stellar brightness across the galaxy. However, a significant portion of these signals originate from eclipsing binary stars, instrument artifacts, or other astrophysical phenomena — not actual planets. Distinguishing real planets from false positives is a critical step in exoplanet research.

Kepler flagged over 9,000 objects of interest during its mission. Each one had to be reviewed and labeled:

- **Confirmed** — a verified exoplanet
- **False Positive** — a mimicking signal, usually an eclipsing binary star or instrument artifact
- **Candidate** — unresolved, awaiting follow-up confirmation

Manual review is expensive and slow. This project automates the triage using machine learning, and for the ~400 unresolved candidates in the dataset, ranks them by how likely they are to be real planets — helping prioritize which ones deserve follow-up attention first.

---

## Dataset

[Kepler Exoplanet Dataset on Kaggle](https://www.kaggle.com/datasets/gauravkumar2525/kepler-exoplanet-dataset) — originally from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)

9,564 observations with 9 orbital and stellar features.

| Label | Class | Description |
|-------|-------|-------------|
| 0 | FALSE POSITIVE | Not a real exoplanet — likely an eclipsing binary or instrument artifact |
| 1 | CANDIDATE | Unresolved — astronomers haven't confirmed or ruled it out yet |
| 2 | CONFIRMED | Verified exoplanet |

---

## What I Did

The dataset had 9 orbital and stellar features, so I started by engineering 5 new ones to capture relationships the raw features couldn't express on their own:

- **`koi_multiplicity`** — counts how many candidates share the same host star. Multi-planet systems carry a strong prior toward authenticity; the probability of multiple independent false signals occurring around the same star by chance is extremely low.
- **`score_sq`** — KOI score squared, to amplify the difference between high and low confidence signals. This turned out to be the single most important feature in the model according to SHAP analysis.
- **`radius_ratio`** — planet radius divided by stellar radius (koi_prad / koi_srad). A planet's absolute size means little without context — this ratio captures the relationship between the two and helps identify eclipsing binaries, which tend to have extreme values.
- **`score_period`** — confidence score multiplied by orbital period, combining two independent signals: a high-confidence, long-period object is a strong planet candidate; a low-confidence, short-period object is more likely a false positive.
- **`temp_ratio`** — equilibrium temperature divided by stellar effective temperature. Unusual ratios can indicate a misidentified signal rather than a real planet.

From there, I handled missing values with iterative imputation, log-transformed right-skewed features, and scaled with RobustScaler (more stable than StandardScaler with outliers). All preprocessing steps are wrapped in a scikit-learn Pipeline to prevent data leakage. SMOTE oversampling was applied post-scaling to address class imbalance.

I then tuned XGBoost with RandomizedSearchCV over 40 iterations using stratified 5-fold cross-validation, and combined it with a Random Forest in a stacking ensemble with a Logistic Regression meta-learner.

---

## Results

### Multiclass Classification (FALSE POSITIVE / CANDIDATE / CONFIRMED)

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Decision Tree | 76% | 0.71 |
| XGBoost (tuned) | 80% | 0.76 |
| Stacking Ensemble | 80% | 0.76 |

The trickiest class was CANDIDATE — and that's not a model failure, that's just reality. These are objects scientists haven't confirmed yet, so no algorithm can reliably classify something that astronomers themselves are still unsure about.

### Binary Classification (FALSE POSITIVE vs CONFIRMED)

To test this, I ran a separate binary classifier on just FALSE POSITIVE vs CONFIRMED, dropping CANDIDATEs entirely. The results confirm that the multiclass ambiguity was coming from the labels, not the model.

| Metric | Score |
|--------|-------|
| Accuracy | 97% |
| ROC-AUC | 0.995 |
| Average Precision | 0.99 |

### Candidate Priority Ranking

Rather than forcing a classification on unresolved candidates, the pipeline assigns each one a confirmation probability and ranks them for follow-up prioritization.

| Priority Tier | Threshold | Count |
|---------------|-----------|-------|
| High priority | P(confirmed) > 0.80 | 42 |
| Likely false positive | P(confirmed) < 0.20 | 281 |
| Genuinely ambiguous | 0.40 < P(confirmed) < 0.60 | 13 |

---

## Astrophysical Interpretation

SHAP analysis identifies the features driving model predictions. The results are consistent with established astrophysical reasoning.

### Multiclass (FALSE POSITIVE / CANDIDATE / CONFIRMED)

- **`score_sq`** — the most important feature overall. Squaring the confidence score amplifies the gap between high and low confidence objects, making the separation between confirmed planets and false positives more distinct.
- **`koi_score`** — Kepler's own instrument confidence score, the second strongest signal. The model's reliance on this validates the approach — it is independently converging on the telescope's own judgment.
- **`koi_prad`** — planetary radius. Real planets fall within a physically constrained size range; objects outside this range are statistically more likely to be false positives.
- **`radius_ratio`** — more discriminative than planetary radius alone. Extreme ratios are characteristic of eclipsing binary stars mimicking planetary transits.
- **`koi_kepmag`** — the brightness of the star as observed by Kepler. Fainter stars produce noisier light curves, which increases the chance of a misidentified signal.
- **`koi_multiplicity`** — stars with multiple candidates are far more likely to host real planets. The model picked this up without being told — it learned the multiplicity boost from the data.
- **`score_period`** — combines confidence and orbital period into one signal. High confidence at long periods strongly favors a real planet.
- **`koi_steff`** — stellar effective temperature. Hotter stars are more active, which can produce false signals that mimic planetary transits.
- **`koi_period`** — short orbital periods are disproportionately associated with false positives, as grazing eclipsing binaries preferentially cluster at short periods.

### Binary (FALSE POSITIVE vs CONFIRMED)

- **`score_sq`** — remains the dominant feature, with an even larger margin than in the multiclass task. With candidates removed, the model leans even harder on the confidence signal.
- **`koi_score`** — second strongest, consistent with the multiclass result.
- **`koi_prad`** — planetary radius plays a larger role here than in multiclass, likely because confirmed planets cluster more tightly in size than the full three-class distribution.
- **`koi_multiplicity`** — jumps in relative importance in the binary task. Without the ambiguous candidate class diluting the signal, the multiplicity boost becomes a cleaner discriminator.
- **`radius_ratio`** — consistent with multiclass. Eclipsing binaries have characteristic radius ratios that the model reliably identifies.
- **`score_period`** — slightly more important in the binary task, reinforcing that the combination of confidence and orbital period is a strong signal for confirmed planets specifically.
- **`koi_steff`** — stellar effective temperature remains relevant, consistent with its role in the multiclass task.
- **`koi_kepmag`** — star brightness continues to contribute, though with less weight than in multiclass.
- **`koi_period`** — short periods remain associated with false positives in the binary task as well.
- **`koi_insol`** — insolation flux appears in the binary top 10, reflecting that the amount of stellar energy a planet receives helps distinguish real orbital configurations from mimicking signals.

---

## Files

```
/kaggle
    exoplanets_data.csv
/models
    preprocessing_pipeline.pkl
    stacking_ensemble.pkl
    xgb_binary.pkl
    xgb_multiclass.pkl
/notebooks
    kepler_koi_classification.ipynb
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
