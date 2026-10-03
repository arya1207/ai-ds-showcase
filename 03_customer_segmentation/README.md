# 03 Customer Segmentation

An unsupervised learning project that finds natural customer groups in credit card data, then checks whether those groups behave differently.

## Business question

What natural customer segments exist, and how do they differ in behavior and risk?

## Data

The same UCI "Default of Credit Card Clients" dataset used in project 2. This notebook reads it from `../02_credit_default_model/data/credit_default.csv`, so download it once as described in the project 2 README.

The default column is not used to build segments. It is used afterward as an outside check.

## Approach

1. Select five behavior features: credit limit, age, credit utilization, log of last payment, and latest payment status.
2. Standardize the features. K-means and PCA depend on distance, so scale matters.
3. Try K-means for k from 2 to 7. Compare silhouette scores and judge how explainable each option is.
4. Fit the final model and profile each segment.
5. Name the segments in plain business language.
6. Plot the segments in 2D with PCA.
7. Run Isolation Forest to flag unusual customers.