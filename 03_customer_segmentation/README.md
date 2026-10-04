# 03 Customer Segmentation

An unsupervised learning project that finds natural groups in credit card customers, then checks whether those groups differ in risk even though the model never saw the default label.

## Business question

What customer segments exist, and how do they differ in behavior and risk?

## Data

The same UCI "Default of Credit Card Clients" dataset used in project 2: 30,000 customers with credit limit, age, payment status, bill amounts, and payment amounts.

Source: I. Yeh, "Default of Credit Card Clients," UCI Machine Learning Repository, 2009. [Online]. Available: https://doi.org/10.24432/C55S3H

The notebook reads the file from `../02_credit_default_model/data/credit_default.csv`, so follow the data steps in the project 2 README first.

The default column is not used to build segments. It is used afterward as an outside check.

## Approach

1. **Select five behavior features:** credit limit, age, credit utilization (latest bill divided by limit), log of the latest payment amount, and latest payment status.
2. **Standardize** the features so each has mean 0 and standard deviation 1. K-means depends on distance, so scale matters.
3. **Try K-means** for 2 to 7 segments and compare silhouette scores.
4. **Fit the final model** and profile each segment.
5. **Validate** with the default rate (not used in clustering) and a stability check across random seeds.
6. **Visualize** the segments in two dimensions with PCA.
7. **Flag unusual customers** with Isolation Forest.

## Results

### Choosing the number of segments

| k | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|
| Silhouette | 0.242 | 0.269 | 0.249 | 0.263 | 0.256 | 0.262 |

Silhouette scores are low and nearly flat, from 0.24 to 0.27. That means customers form a continuum more than sharply separated clusters, which is common for behavioral data. K = 3 scored highest by a small margin. I chose **k = 4** because it gave segments I could explain in business terms. The difference between choices is within the noise of a sampled score.

### The four segments

| Segment | Name | Share | Median limit | Median age | Median utilization | Median pay status | Default rate |
|---|---|---|---|---|---|---|---|
| 0 | Reliable high-limit payers | 32% | 260,000 | 35 | 0.03 | -1 (paid on time) | **11%** |
| 1 | Younger, heavy credit users | 32% | 80,000 | 28 | 0.74 | 0 | **24%** |
| 2 | Established, heavy credit users | 18% | 80,000 | 46 | 0.81 | 0 | **26%** |
| 3 | One month late | 17% | 140,000 | 34 | 0.01 | 1 (one month late) | **36%** |

The overall default rate is 22%. Default rates range from 11% to 36%, a threefold spread, even though the model never used the default column. That suggests the segments pick up real risk patterns.

Segments 1 and 2 have similar risk (24% and 26%). They differ mainly in age, so a three-segment solution would likely merge them. That matches the slightly higher silhouette score at k = 3.

### Stability

Re-running K-means with a different random seed gave an adjusted Rand index of **0.994** against the original segments, where 1.0 means identical. The segments do not depend on the random start.

### PCA view

Two PCA components keep 57% of the variation in the five features (34% and 23%). Segments overlap in the 2D picture, which is expected when five dimensions are drawn in two.

### Unusual customers

Isolation Forest with contamination 0.02 flagged 600 customers. The share is set by that parameter, so 2% is a choice and not a finding. What is informative is how the flagged group differs:

| Group (averages) | Default rate | Credit limit | Age | Utilization | Latest payment status |
|---|---|---|---|---|---|
| Not flagged | 0.21 | 166,636 | 35 | 0.42 | -0.06 |
| Flagged | 0.53 | 209,050 | 45 | 0.65 | 1.98 |

Flagged customers average about two months of payment delay, very small latest payments, and a default rate of 53% versus 21%.

## Key takeaways

- Unsupervised results need business judgment. The silhouette score suggested no clear winner, so interpretability decided k.
- Segments built only from behavior still separate customers by default risk, from 11% to 36%.
- The "one month late" segment has the highest risk and is the natural first group for outreach. The "reliable high-limit payers" segment has about half the average risk.
- The anomaly group is small but about 2.5 times as likely to default as everyone else.

## Limitations

- Low silhouette scores mean segment boundaries are soft. Customers near a boundary could reasonably sit in either group.
- K-means assumes roughly round clusters of similar size, and it is sensitive to outliers.
- Results depend on my choice of five features and on scaling.
- The data is one snapshot from one market and period.
- Median payment status takes whole-number values, so segment profiles are approximate.

## Next steps

- Compare a three-segment solution, Gaussian mixtures, or DBSCAN
- Add more features, such as payment trends over six months
- Combine segments with the project 2 model, for example segment-specific thresholds
- Track segment membership over time to see how customers move between groups

## Run it

From the repo root, with the virtual environment active, open `03_customer_segmentation/segmentation.ipynb` in VS Code and choose Restart and Run All.