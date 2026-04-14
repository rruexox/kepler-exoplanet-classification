# Kepler Exoplanet Classification

Machine learning pipeline to classify exoplanet candidates from NASA's Kepler  
Space Observatory dataset into three categories: FALSE POSITIVE, CANDIDATE, and CONFIRMED.

## Dataset
[Kepler Exoplanet Dataset](https://www.kaggle.com/datasets/gauravkumar2525/kepler-exoplanet-dataset) — 
9,564 observations, 9 features including orbital period, planetary radius, 
equilibrium temperature, insolation flux, and stellar properties.

| Label | Class | Description |
|-------|-------|-------------|
| 0 | FALSE POSITIVE | Not a real exoplanet |
| 1 | CANDIDATE | Potential exoplanet, awaiting confirmation |
| 2 | CONFIRMED | Verified exoplanet |

## Approach
1. Data cleaning and train/test split
2. Feature engineering (score-period interaction, radius ratio, temperature ratio)
3. Log transforms + IterativeImputer + RobustScaler
4. SMOTE for class imbalance
5. Baseline: Decision Tree
6. Primary: XGBoost with RandomizedSearchCV tuning
7. Ensemble: Soft Voting (XGBoost + Random Forest)
8. Binary classification: FALSE POSITIVE vs CONFIRMED only

## Results

### 3-Class Classification
| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Decision Tree | 0.76 | 0.71 |
| XGBoost (tuned) | 0.80 | 0.76 |
| Ensemble | 0.81 | 0.77 |

### Binary Classification (FALSE POSITIVE vs CONFIRMED)
| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| XGBoost | 0.98 | 0.97 |

## Key Finding
The CANDIDATE class is inherently difficult to classify — these are objects 
that scientists themselves have not yet confirmed or denied. When excluding 
CANDIDATE and running a binary classification between FALSE POSITIVE and 
CONFIRMED, the model achieves 98% accuracy, demonstrating that the pipeline 
is highly effective for the cases where a definitive label exists.

## Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
```