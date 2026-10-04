# 02 Credit Default Model

A supervised learning project that predicts which credit card customers will miss their next payment, and chooses a decision threshold for a realistic outreach goal.

## Business question

Which customers are likely to default next month, so outreach can focus on them first?

## Data

UCI "Default of Credit Card Clients" dataset: 30,000 customers and 23 input columns covering credit limit, demographics, 6 months of payment status, bill amounts, and payment amounts. The target is whether the customer defaulted the following month. About 22% of customers defaulted.

Source: I. Yeh, "Default of Credit Card Clients," UCI Machine Learning Repository, 2009. [Online]. Available: https://doi.org/10.24432/C55S3H. Downloaded from https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

The data file is not committed. To run the notebook:

1. Download the original Excel file from the UCI repository.
2. Remove the first label row so the real column names become the header row.
3. Save it as `data/credit_default.csv` in this folder.

## Approach

1. **Clean.** Map undefined category codes in education and marriage to "other".
2. **Engineer two features.** Credit utilization (latest bill divided by credit limit) and average payment delay across 6 months.
3. **Split.** Stratified 80/20 train and test split, so both sets keep the 22% default rate. The test set is used once, for the final score.
4. **Compare three models** with 5-fold stratified cross-validation on ROC-AUC:
   - Logistic regression (scaled, class-weighted). This is the simple baseline.
   - Random forest (class-weighted).
   - Gradient boosting (`HistGradientBoostingClassifier`).
5. **Evaluate** the best model on the test set with ROC-AUC, PR-AUC, precision, and recall.
6. **Tune the threshold** for the business goal.
7. **Explain the drivers** with permutation importance.

All random seeds are fixed, so rerunning the notebook gives the same numbers.

## Results

| Model | CV ROC-AUC (5-fold, mean +/- SD) |
|---|---|
| Logistic regression (baseline) | 0.727 +/- 0.010 |
| Random forest | 0.772 +/- 0.005 |
| **Gradient boosting** | **0.784 +/- 0.004** |

Test set, gradient boosting: ROC-AUC **0.778**, PR-AUC **0.558**. The no-skill PR-AUC equals the default rate, 0.221, so the model is about 2.5 times better than guessing on that measure. Test AUC is close to the cross-validation score, which suggests the model is not overfitting.

At the default 0.5 threshold, the model reaches 66% precision and 37% recall on defaulters. It misses most of them, so I tuned the threshold.

### Threshold choice

Lift is precision divided by the 22% base rate: how much more likely a flagged customer is to default than a random customer.

| Threshold | Precision | Recall | Share flagged | Lift |
|---|---|---|---|---|
| 0.2 | 0.42 | 0.67 | 35% | 1.9x |
| **0.3** | **0.55** | **0.54** | **22%** | **2.5x** |
| 0.4 | 0.62 | 0.43 | 15% | 2.8x |
| 0.5 | 0.66 | 0.37 | 12% | 3.0x |

I would use **0.3**. It catches about half of defaulters while contacting 22% of customers. A missed default costs more than an extra outreach call, so I favor recall over the default cutoff. If outreach is very cheap, 0.2 raises recall to 67% at the cost of precision.

### What drives the model

Permutation importance shows the drop in test AUC when each feature is shuffled. The latest payment status (`PAY_0`) has the largest drop. This matches the exploratory table, where default rates rise from about 13% for on-time payers to about 69% for customers two months late.

## Key takeaways

- About 22% of customers default, so I compared models with AUC and chose a threshold with precision and recall. Accuracy would mislead: a model that predicts "no default" for everyone scores 78%.
- Gradient boosting beat the logistic baseline by about 0.06 AUC. The gap is many times larger than the fold-to-fold variation.
- Recent payment behavior is the strongest signal.
- The right threshold depends on the cost of a missed default versus an outreach call, which is a business decision and not a model property.

## Limitations

- The threshold table was computed on the test set. A cleaner approach would choose the threshold on a separate validation set and report test results once.
- Payment-status groups with extreme delays are tiny (fewer than 100 customers each), so their default rates are noisy.
- Correlated features, such as `PAY_0` and average payment delay, share importance, so individual importance values should be read loosely.
- The data is one snapshot from one market and period. If this were a time-based problem, I would split by date to avoid leakage.
- No hyperparameter tuning, probability calibration, or fairness review.

## Next steps

- Choose the threshold on a validation set and test it across cost assumptions
- Compare XGBoost or LightGBM with tuned hyperparameters
- Add SHAP explanations for individual customers
- Calibrate probabilities
- Serve the model behind an API, log predictions, and monitor drift

## Run it

From the repo root, with the virtual environment active, open `02_credit_default_model/model.ipynb` in VS Code and choose Restart and Run All.