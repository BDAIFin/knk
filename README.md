# Fraud Detection System (FDS) – Stage 1 Baseline Pipeline

## 1. Overview

This project implements a production-oriented fraud detection pipeline designed to mimic real-world Financial Detection Systems (FDS).

The goal of Stage 1 is not only to maximize predictive performance, but to:

* Preserve realistic class distribution
* Maintain operational interpretability
* Evaluate recall–precision trade-offs under deployment constraints
* Avoid unnecessary information loss

The system is structured around a time-aware data split and a cost-sensitive logistic regression baseline.

---

## 2. Data Splitting Strategy

Data is strictly split by chronological order:

* **Train**: Historical period (model fitting only)
* **Test**: Future period (performance evaluation)
* **Check**: Later unseen period (operational simulation / drift inspection)

No sampling or transformation is performed before time-based splitting to prevent leakage.

All imbalance handling is applied **only to the training set**.

---

## 3. Feature Engineering

Feature construction focuses on behavioral and risk-relevant signals rather than raw transaction values.

### Transaction-Level Features

* `amount`
* `is_refund`
* `log_abs_amount`
* `amount_income_ratio`
* `amount_limit_ratio`

### Customer & Risk Profile

* `credit_score`
* `total_debt`
* `num_credit_cards`
* `credit_limit`
* `years_since_pin_change`
* `years_to_retirement`

### Behavioral & Contextual

* Time features (`tx_year`, `tx_month`, `tx_hour`, `is_weekend`)
* Merchant category one-hot encoding
* Distance from home location
* Income normalization features

All features are numerical and suitable for logistic regression modeling.

---

## 4. Imbalance Handling Strategy

Dataset imbalance ratio (train):

```
fraud:      6,598
non-fraud: 603,057
≈ 1 : 91
```

Rather than blindly applying sampling, the following structured procedure was followed:

1. Train baseline model on raw distribution.
2. Apply cost-sensitive learning (`class_weight="balanced"`).
3. Compare with:

   * Random UnderSampling
   * Random OverSampling
4. Evaluate all models on the original test distribution.

### Key Finding

Sampling (under/over) did **not** improve PR-AUC meaningfully.

Under-sampling introduces potential information loss.
Over-sampling duplicates minority patterns without adding new information.

Final decision:

> Use **raw training data without sampling**.

---

## 5. Model – Stage 1 Baseline

Model:

* Logistic Regression
* StandardScaler preprocessing
* Cost-sensitive learning evaluated

Metrics:

* PR-AUC (primary metric)
* ROC-AUC (secondary metric)
* Precision at Recall ≥ 0.70
* Alert Rate
* Confusion Matrix

---

## 6. Threshold Selection Strategy

Instead of using a default 0.5 threshold, we select threshold based on an operational constraint:

> Maximize precision subject to Recall ≥ 0.70

This simulates a real-world detection policy where recall target is fixed and alert volume must be controlled.

Threshold is selected from the Precision–Recall curve.

---

## 7. Experimental Comparison

Models compared:

* Raw + no class_weight
* Raw + class_weight
* Under-sampled + no class_weight
* Under-sampled + class_weight
* Over-sampled + no class_weight
* Over-sampled + class_weight

Result summary:

* Raw training distribution achieved the highest PR-AUC.
* Sampling did not provide consistent gains.
* Cost-sensitive learning reduced alert rate slightly.
* No significant advantage from under/over sampling.

Final selection: **Raw distribution baseline model**.

---

## 8. Operational Considerations

* No artificial distortion of test distribution.
* No unnecessary loss of normal transaction diversity.
* Threshold selection aligned with deployment constraints.
* Sampling treated as experimental comparison, not default solution.

This reflects real-world FDS development practices where stability and interpretability matter as much as performance.

---

## 9. Project Structure

```
project/
│
├── data/
│   ├── train.parquet
│   ├── test.parquet
│   └── check.parquet
│
├── notebooks/
│   ├── 01_feature_engineering.ipynb
│   ├── 02_baseline_model.ipynb
│   └── 03_sampling_experiments.ipynb
│
├── models/
│   └── stage1_logistic.py
│
└── README.md
```

---

## 10. Future Work

* Time stability analysis (year-wise evaluation)
* Alert-rate constrained optimization
* Stage 2 anomaly modeling
* Distribution shift monitoring
* Model calibration

