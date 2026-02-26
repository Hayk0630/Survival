# In-Hospital Mortality Prediction

This project predicts whether a patient will die during hospital stay (`In-hospital_death`) using clinical data.

## Objective
Build a practical ML pipeline that prioritizes recall while still considering precision.  
Final scoring emphasis used in experiments:

- $0.8 \times Recall + 0.2 \times Precision$

## Dataset
- File: `Survival_dataset.csv`
- Target: `In-hospital_death`
- The dataset is included in this repository for reproducibility.

## What was tried
We tested and compared:
- Logistic Regression
- Random Forest
- Gradient Boosting (selected as main model)
- XGBoost
- Additional feasibility checks with ExtraTrees / HistGradientBoosting / LightGBM

### Main chosen model
`GradientBoostingClassifier` with:
- `n_estimators=100`
- `learning_rate=0.1`
- `max_depth=3`
- `min_samples_leaf=1`
- Decision threshold: `0.20`

## Results (comparison table)
At fixed threshold `0.20`, we compared the selected model with other strong baselines.

![Model comparison](comparison_table.png)

From the table:
- **GradientBoosting** gives the strongest recall among compared models at this threshold.
- **XGBoost** gives higher precision but lower recall than GradientBoosting.

## Key takeaway
With current feature set and label definition, very high recall is achievable, but precision remains limited.  
This suggests performance is currently constrained more by signal/data quality than by model family alone.

## Repository artifacts
- `getting_the_data.py` -> fitted preprocessing pipeline logic
- `model.py` -> production model class (Gradient Boosting + thresholded prediction)
- `preprocessor.joblib` -> saved preprocessor
- `model.joblib` -> saved trained model
- `model.ipynb` -> full experiment notebook and visual outputs

## Quick inference (without retraining)
```python
import pandas as pd
from getting_the_data import Preprocessor
from model import Model

pre = Preprocessor.load('preprocessor.joblib')
mdl = Model.load('model.joblib')

df = pd.read_csv('Survival_dataset.csv')
X = df.drop(['Length_of_stay', 'Survival', 'SAPS-I', 'SOFA', 'In-hospital_death'], axis=1)

X_proc = pre.transform(X.head(5))
predictions = mdl.predict(X_proc)
probabilities = mdl.predict_proba(X_proc)

print(predictions)
print(probabilities)
```
