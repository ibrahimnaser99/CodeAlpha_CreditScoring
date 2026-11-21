# CodeAlpha Credit Scoring

Credit default prediction project using the UCI Credit Card Default dataset.

## Dataset

The inspection script downloads dataset 350 directly from the UCI Machine
Learning Repository when run; the dataset is not stored in or uploaded with
this repository. Any local `data/` directory is ignored by Git.

Install the project dependencies and inspect the data:

```bash
python -m pip install -r requirements.txt
python scripts/inspect_data.py
python -m unittest discover -s tests -t . -v
python eda.py
python prepare_data.py
python train_baselines.py
python tune_models.py
python select_final.py --model "Random Forest" --cost-ratio 3
python explain_model.py
```

### Data inspection and cleaning

The dataset contains 30,000 rows, 23 feature columns, and one target column.
The initial target share for defaults (`Y = 1`) is 22.12%, and the feature
table has no missing values. There are 56 duplicated feature rows, but only 35
exact duplicate rows when the target is included; the latter are removed
before splitting the data. The inspection script prints the UCI variable
descriptions, target distribution, and low-cardinality feature values.

The UCI variable descriptions confirm the positional mapping: X1 is
`LIMIT_BAL`, X2-X5 are sex, education, marital status, and age; X6-X11 are
repayment status from September through April 2005; X12-X17 are the matching
monthly bill amounts; and X18-X23 are the matching previous payment amounts.
Education codes 1-4 and marriage codes 1-3 are documented by UCI. Other
education and marriage codes are grouped as "others". The UCI description
documents repayment status values -1 and 1-9, but the data also contains -2
and 0; those observed codes are retained unchanged, without inferring their
meaning.

The EDA cleaning report on the fetched dataset showed 30,000 rows before
cleaning, 35 exact duplicates removed, 345 education codes remapped, and 54
marriage codes remapped, leaving 29,965 rows. The cleaned default share is
22.13% (6,630 clients). Among `pay_1` groups, the observed default rates range
from 12.8% for code 0 (14,737 clients) to 77.8% for code 7 (9 clients); rates
for very small groups should be interpreted cautiously.

`python eda.py` saves four figures under `reports/figures/`: target class
counts, default rate by `pay_1`, credit-limit distributions by outcome, and a
numeric correlation matrix. The class-count and repayment-status figures
clearly show imbalance and the increasing default rate for delayed repayment.
The credit-limit distributions overlap substantially and are right-skewed;
the correlation plot is readable but dense, and correlations should not be
interpreted as causal effects.

### Feature engineering and data splits

`preprocessing.py` creates six features using each client's own six-month
history, then performs a stratified 60/20/20 train/validation/test split with
seed 42. The row-wise features do not learn from other rows, so they can be
computed before splitting. Any learned preprocessing (such as scaling,
encoding, or imputation) must be fit on training data only, preferably within
a model pipeline. Use validation data for model and threshold selection; keep
the test set for the final evaluation only.

Feature definitions and handling decisions:

- `avg_utilization` and `max_utilization`: monthly bill divided by credit
  limit, averaged or maximized across six months. Negative bills are treated
  as zero owed; utilization is not capped and can exceed 1.
- `payment_ratio`: total paid divided by the sum of nonnegative bills, capped
  at 1. If all bills are zero, the ratio is 1.
- `n_late_months`, `max_delay`, and `avg_delay`: count of months with positive
  status codes, maximum delay, and the six-month average delay (non-delay months
  count as zero). Codes 0, -1, and -2 count as no delay; no additional meaning
  is inferred for 0 or -2.

The target and `sex` are excluded from model input features. `sex` is excluded
as a deliberate fairness-related modeling choice.

### Baseline models and evaluation

`modeling.py` builds Dummy (prior), Logistic Regression, Decision Tree, and
Random Forest classifiers. Each model is a scikit-learn `Pipeline`; education
and marriage use one-hot encoding, numeric features are scaled only for
Logistic Regression, and balanced class weights are used for the non-dummy
models. The tree depth and leaf size are constrained.

`train_baselines.py` compares models with stratified 5-fold cross-validation
on the training set and evaluates them on validation. The held-out test split
is not used in this baseline run. It reports ROC-AUC and PR-AUC, precision,
recall, F1, and confusion-matrix counts at threshold 0.5. It also reports
validation metrics at the threshold that maximizes validation F1; that
threshold and its validation F1 are optimistic because they are selected and
measured on the same validation data. The script saves the metrics table to
`reports/baseline_results.csv` and ROC/precision-recall plots to
`reports/figures/baselines_validation_curves.png`.

The current threshold rule is maximum F1 on validation, which balances
precision and recall but does not encode specific business costs. A false
negative is a defaulter not flagged; a false positive is a non-defaulter
flagged. Select an operational threshold only after deciding how those costs
should be balanced, and reserve the test split for one final evaluation.

### Hyperparameter tuning

`tune_models.py` runs randomized hyperparameter searches for Logistic
Regression, Decision Tree, Random Forest, and Hist Gradient Boosting. Each
candidate is a pipeline, and the 5-fold stratified cross-validation runs on
the training split only; preprocessing is refit inside every fold. The
combination with the best mean PR-AUC is selected. Validation is used only to
report a separate ranking score for the selected estimator; the test split
remains unused. By default, the search tries 20 parameter combinations per
model (`--n-iter` can change this; `--quick` tries 3 for a smoke test).

The script saves CV and validation scores to `reports/tuning_results.csv`
and the selected parameters to `reports/best_params.json`.

The randomized search found some best values at or near the edges of the
searched ranges (`C` near its lower bound for Logistic Regression,
`min_samples_leaf` near its upper bound for Decision Tree, and
`max_features=0.5` at the upper categorical choice for Random Forest). This
means a better value could exist outside those ranges; however, the observed
improvement was small and further tuning is intentionally out of scope.

### Final model decision, threshold, and interpretation

Before any test evaluation, the final model is set to **Random Forest**. It
slightly led on validation (PR-AUC 0.567, ROC-AUC 0.788) and was also the
strongest baseline. Hist Gradient Boosting was effectively tied, but the
Random Forest is selected as the tie-breaker; this decision must not be changed
based on test results.

`select_final.py` fits the selected model on training data, calibrates scores
with 5-fold cross-validation on training data by default, then chooses a
threshold using validation and the configured false-negative to false-positive
cost ratio. The preselected ratio for this run is **3:1**. The cost scenarios
1:1, 2:1, 3:1, and 5:1 are illustrative assumptions, not real bank costs.
Calibration is used because balanced class weights can distort raw scores;
without calibration, scores should not be presented as real default
probabilities. The optional `--no-calibration` flag uses raw scores instead.
Validation calibration metrics and the reliability diagram are descriptive,
not independent final performance estimates.

The chosen model and threshold configuration are saved under `models/` and
`reports/`. `explain_model.py` reports validation permutation importance for
the selected model and coefficients for the tuned Logistic Regression.
Correlated repayment and bill-history columns can share permutation
importance, making each individual column appear less important than the
feature group. Logistic Regression coefficients are scaled and strongly
regularized by its tuned low `C`; neither coefficients nor importance imply
causation.

#### Validation decision and test protocol

The selected model is the calibrated Random Forest, with cost ratio 3:1 and
validation threshold **0.25**. At that threshold, validation precision is
0.523, recall is 0.590, and the confusion counts are TN=3,953, FP=714,
FN=543, TP=783. Under the illustrative cost assumptions, validation cost is
0.391 per client versus 0.664 for approving everyone. These are validation
estimates and hypothetical cost units, not realized financial savings or the
project's final test results.

The model, calibration, threshold and cost ratio were chosen on the validation
set and documented before the test set was scored. The test set was scored
once. This order was self-enforced and not independently verifiable.
`evaluate_test.py` refuses to run again after a result is saved unless
`--force` is provided; do not use `--force` for model iteration. The
`models/` directory is ignored by Git because it contains locally generated
model artifacts. Only load joblib artifacts that you created and trust:
loading an untrusted joblib file can execute code.

### Final test results

The locked calibrated Random Forest was evaluated once on the held-out test
split with threshold 0.25 and the illustrative 3:1 false-negative to
false-positive cost ratio.

| Metric | Test result |
|---|---:|
| ROC-AUC (95% bootstrap CI) | 0.776 (0.762–0.790) |
| PR-AUC (95% bootstrap CI) | 0.561 (0.533–0.587) |
| Precision | 0.505 |
| Recall (95% bootstrap CI) | 0.572 (0.545–0.600) |
| F1 | 0.537 |
| Confusion matrix (TN / FP / FN / TP) | 3,923 / 744 / 567 / 759 |
| Cost per client (illustrative units) | 0.408 |
| Approve everyone (same assumed costs) | 0.664 |
| Reject everyone (same assumed costs) | 0.779 |

Validation versus test scores were similar: ROC-AUC 0.787 vs 0.776,
PR-AUC 0.564 vs 0.561, precision 0.523 vs 0.505, recall 0.590 vs 0.572,
and illustrative cost per client 0.391 vs 0.408. These cost values are
hypothetical comparisons, not measured savings or estimates of real lender
losses. Against the illustrative "approve everyone" cost, the test cost is
about 39% lower; this is not a real-world savings estimate. The full command
output is saved in `reports/final_test_output.txt`;
machine-readable metrics and confidence intervals are in
`reports/final_test_results.json`.

### Limitations

The dataset describes credit-card clients in Taiwan in 2005, so results may
not generalize to other populations, periods, products, or lending policies.
The target is the dataset's recorded next-month default label, not a guarantee
of an individual's future default probability. The 3:1 false-negative to
false-positive cost ratio is illustrative rather than based on a lender's
measured losses. Excluding `sex` does not guarantee fairness: age, education,
and marital status remain available and may act as proxies. No subgroup
fairness evaluation was performed. This is an educational project and is not
validated or suitable for real credit decisions.

### Citation

Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques
for the predictive accuracy of probability of default of credit card clients.
*Expert Systems with Applications, 36*(2), 2473–2480.
https://doi.org/10.1016/j.eswa.2007.12.020

Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine
Learning Repository. https://doi.org/10.24432/C55S3H
