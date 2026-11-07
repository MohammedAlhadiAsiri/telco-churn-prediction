# Telco Customer Churn Prediction

Binary classification project that predicts whether a telecom customer will churn, comparing Random Forest, SVM, Naive Bayes, and a soft-voting ensemble, with decision-threshold tuning.

**Notebook:** [`telco_churn_prediction.ipynb`](telco_churn_prediction.ipynb)
**Dataset:** [Telco Customer Churn (IBM sample data) on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Not included in this repo; download it and place it where the notebook expects it.

## Problem

Retaining a customer is usually cheaper than acquiring a new one. Given a customer's account, services, and billing information, can we flag who is likely to leave so a retention offer can be targeted?

- **Target:** `Churn` (Yes/No), encoded as 1/0
- **Size:** about 7,000 customers, 20 features (demographics, services, contract, billing)
- **Class balance:** about 27% churn, so accuracy alone is misleading. A model that always predicts "no churn" scores about 73% accuracy and catches zero churners.

## Data cleaning

- `TotalCharges` loads as text because 11 rows contain a blank string. After converting to numeric, all 11 rows have `tenure == 0` (brand-new customers not yet billed), so they were filled with 0.
- Dropped `customerID` (identifier, no predictive value).
- One-hot encoded categorical features (`drop_first=True`).
- Stratified 80/20 train/test split (`random_state=42`).

## Key EDA findings

- **Low tenure means higher churn.** Customers tend to leave early in their relationship.
- **Contract type matters.** Month-to-month customers churn far more than one- or two-year contracts.
- **Fiber optic subscribers had the highest churn rate** among internet service types. Possible explanations (untested): higher monthly prices, more month-to-month contracts, service quality or competition not captured in the data.
- **Higher monthly charges go with higher churn**, but part of this reflects which services a customer has, not price alone.
- **`TotalCharges` is largely redundant with `tenure`** (roughly `tenure x MonthlyCharges`), so the "low total charges means churn" pattern is mostly the tenure effect again. This matters for interpreting linear model coefficients.
- `MonthlyCharges` is bimodal (a spike near the low end, likely customers without internet service, and a broad hump at higher values), not normally distributed.

## Approach

| Step | Details |
|---|---|
| Baseline | `DummyClassifier(strategy="most_frequent")`: 0.73 accuracy, 0 recall on churners |
| Models | Random Forest, SVM (RBF kernel), Gaussian Naive Bayes, soft-voting ensemble of the three |
| Preprocessing | `StandardScaler` inside pipelines for SVM and Naive Bayes, so scaling is fit only on training folds (no leakage) |
| Imbalance | `class_weight="balanced"` for Random Forest and SVM |
| Evaluation | 5-fold stratified cross-validation on the training set; test set used once at the end |

## Results (5-fold CV on the training set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Voting (soft) | 0.749 | 0.519 | 0.779 | 0.623 | 0.836 |
| Random Forest | 0.791 | 0.647 | 0.473 | 0.546 | 0.828 |
| SVM | 0.741 | 0.509 | 0.781 | 0.616 | 0.824 |
| Naive Bayes | 0.664 | 0.434 | 0.878 | 0.581 | 0.822 |

**Takeaways**

- All four models rank customers by churn risk about equally well (ROC-AUC 0.82 to 0.84). The differences in accuracy, precision, and recall mostly reflect where each model sits at the default 0.5 threshold, not how well it separates the classes.
- Random Forest has the best accuracy and precision but misses more than half of churners at 0.5.
- Naive Bayes catches the most churners (recall 0.88) but over-predicts churn, and its accuracy falls below the dummy baseline.
- The voting ensemble gives the best ROC-AUC and a balanced middle ground, though its edge over the single models is small.

## Threshold tuning and test-set results

The decision threshold was chosen from out-of-fold predictions on the training set (maximizing F1), then applied once to the held-out test set.

- Chosen threshold: 0.557 (about 0.56)
- Test ROC-AUC (voting ensemble): 0.829, consistent with the 0.836 from cross-validation, so there is no sign of overfitting to the training folds.

| Threshold | Precision (churn) | Recall (churn) | F1 (churn) | Accuracy |
|---|---|---|---|---|
| 0.50 | 0.51 | 0.77 | 0.62 | 0.75 |
| 0.56 (tuned) | 0.54 | 0.72 | 0.62 | 0.77 |

Raising the threshold from 0.50 to 0.56 trades some recall (0.77 to 0.72) for precision (0.51 to 0.54), and F1 for the churn class stays at 0.62. At 0.50 the model catches about 77% of churners but only about half of the customers it flags actually leave. At 0.56 it catches 72% with slightly fewer false alarms. Because F1 is flat, the choice between the two comes down to the cost trade-off below rather than to the metric.

**Business trade-off:** if a missed churner costs more than a wasted retention offer, favor a lower threshold (higher recall). If offers are expensive, favor a higher threshold (higher precision).

## Limitations

- Single dataset snapshot with no time dimension, so the model can't capture changes in behavior over time.
- Features such as service quality, customer support contacts, and competitor offers are not available.
- Naive Bayes assumes feature independence, which one-hot encoded and correlated features violate.
- Results come from one train/test split; confidence intervals were not computed.

## Next steps

- Add a logistic regression benchmark and compare coefficients with Random Forest feature importances.
- Error analysis: which customer segments does the model miss most?
- Try gradient boosting (XGBoost or LightGBM) and hyperparameter tuning.
- Move cleaning and feature engineering into a reusable script that outputs Parquet.

## How to run

1. Download the dataset from Kaggle and place the CSV where the notebook expects it (or mount Google Drive in Colab and update the path).
2. Install dependencies: `pip install -r requirements.txt`
3. Open the notebook in Jupyter or Google Colab and run all cells.

## Tech stack

Python, pandas, NumPy, scikit-learn, seaborn, matplotlib

## License

MIT
