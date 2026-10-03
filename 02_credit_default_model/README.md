# 02 Credit Default Model

A supervised learning project that predicts which credit card customers will miss their next payment.

## Business question

Which customers are likely to default next month, so outreach can focus on them first?

## Data

UCI "Default of Credit Card Clients" dataset: about 30,000 customers, 23 features covering credit limit, demographics, 6 months of payment status, bill amounts, and payment amounts. The target is whether the customer defaulted the following month.

The data file is not committed. To run the notebook:

1. Download the dataset from the UCI Machine Learning Repository (or a Kaggle copy).
2. Save it as `data/credit_default.csv` in this folder.

## Approach

1. Clean undefined category codes in education and marriage.
2. Engineer two features: credit utilization (latest bill divided by limit) and average payment delay across 6 months.
3. Stratified 80/20 train and test split. The test set is used once at the end.
4. Compare three models with 5-fold stratified cross-validation on ROC-AUC:
   - Logistic regression (scaled, class-weighted baseline)
   - Random forest (class-weighted)
   - Gradient boosting (`HistGradientBoostingClassifier`)
5. Evaluate the best model on the test set: ROC-AUC, PR-AUC, precision, recall.
6. Tune the decision threshold for the business goal.
7. Explain drivers with permutation importance.