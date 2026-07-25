# Off-Spec Risk Model — Evaluation Report

**Selected model:** Histogram Gradient Boosting  
**Selection criterion:** highest validation pr_auc, tie-broken on median_warning_min  
**Operating threshold:** 0.8500 (chosen on validation via `f1`)  
**Prediction target:** basis weight deviates more than 2.5% from setpoint within the next 10 minutes

## Leakage controls

- Splits are **event-wise**, never row-wise.
- Features are strictly backward-looking (asserted by `test_features.py::test_no_future_leakage`).
- Threshold selected on validation only.
- Model selection used validation only.
- **Test split scored exactly once**, after the winner was fixed.
- Permutation importance computed on validation, not training.

## Operating point

Alarm on-delay: **1 samples (5 s)** -- an alarm is confirmed only after the risk score stays above threshold for that long (ISA-18.2 on-delay).

The threshold and on-delay are tuned on an **event-level** objective, not row-level F1. A transition is ~336 samples long, so even a 6% row-level false positive rate makes a nuisance alarm near-certain in every clean transition, and a system that does that gets switched off. Conversely row-level recall understates usefulness: an excursion only has to be caught once to warn the operator. The objective is therefore to minimise per-clean-event false alarms subject to a floor on event detection rate.

## Evaluation population

Classification metrics are computed on rows where basis weight is still **inside** the ±2.5% band. Predicting a breach while the sheet is already off-spec is trivial and would inflate every score. Early-warning metrics necessarily span all rows, since they measure the gap to the breach itself.

## Split composition

| split   |   events |   rows |   positive_rate |   in_spec_rows |   in_spec_positive_rate |
|:--------|---------:|-------:|----------------:|---------------:|------------------------:|
| train   |      908 | 305088 |        0.304624 |         272056 |                0.22234  |
| val     |      296 |  99456 |        0.31059  |          89576 |                0.236704 |
| test    |      296 |  99456 |        0.316673 |          88095 |                0.230717 |

## Model comparison (validation)

| model                  |   pr_auc |   precision |   recall |       f1 |       fpr |   median_warning_min |   mean_warning_min |   detection_rate |   false_alarm_event_rate |
|:-----------------------|---------:|------------:|---------:|---------:|----------:|---------------------:|-------------------:|-----------------:|-------------------------:|
| random_forest          | 0.784813 |    0.882332 | 0.47814  | 0.620194 | 0.0197739 |              4.58333 |            4.90045 |         0.805405 |                0.108108  |
| hist_gradient_boosting | 0.782351 |    0.881245 | 0.443428 | 0.589985 | 0.0185307 |              4.41667 |            4.56556 |         0.810811 |                0.0720721 |
| logistic_regression    | 0.75919  |    0.821318 | 0.499693 | 0.621353 | 0.0337121 |              4.875   |            5.66009 |         0.821622 |                0.315315  |

## Selected model — validation

| Metric | Value |
|---|---|
| PR-AUC | 0.7824 |
| Precision | 0.8812 |
| Recall | 0.4434 |
| F1 | 0.5900 |
| False positive rate | 0.0185 |
| Median warning time | 4.42 min |
| Mean warning time | 4.57 min |
| **Event detection rate** | 0.811 |
| False alarm rate (per clean event) | 0.072 |

## Selected model — test (held out, scored once)

| Metric | Value |
|---|---|
| PR-AUC | 0.8267 |
| Precision | 0.8945 |
| Recall | 0.4938 |
| F1 | 0.6363 |
| False positive rate | 0.0175 |
| Median warning time | 4.67 min |
| Mean warning time | 5.03 min |
| **Event detection rate** | 0.827 |
| False alarm rate (per clean event) | 0.099 |

## Explainability

SHAP method: `shap.TreeExplainer` (exact=True), 3000 rows explained.

### Top 15 features by consensus attribution

| feature                         |   permutation_norm |   shap_norm |   consensus |
|:--------------------------------|-------------------:|------------:|------------:|
| bw_dev_headroom_pct             |         0.319711   |   0.124664  |   0.222188  |
| t_since_ramp_min                |         0.141165   |   0.1397    |   0.140433  |
| transition_magnitude            |         0.128504   |   0.064181  |   0.0963426 |
| plan_trim_enabled               |         0.0763791  |   0.0319791 |   0.0541791 |
| plan_ramp_min                   |         0.0679219  |   0.0346192 |   0.0512705 |
| ramp_progress                   |         0.0412249  |   0.0499401 |   0.0455825 |
| mv_stock_flow_remaining_frac    |         0.0263263  |   0.0414542 |   0.0338902 |
| plan_tau_c_scale                |         0.0168594  |   0.0419169 |   0.0293881 |
| bw_abs_dev_pct                  |         0.0167861  |   0.0368131 |   0.0267996 |
| dv_headbox_consistency_roc_2min |         0.026566   |   0.0146138 |   0.0205899 |
| dv_headbox_consistency          |         0.020477   |   0.018408  |   0.0194425 |
| plan_lead_scale                 |        -0.00102816 |   0.0352401 |   0.017106  |
| mv_stock_flow_remaining         |         0.0104     |   0.0189129 |   0.0146565 |
| bw_change_pct                   |         0.00648312 |   0.020765  |   0.013624  |
| mv_machine_speed_remaining_frac |         0.00589585 |   0.0206077 |   0.0132518 |

Consensus averages the model's native importance, permutation importance on validation, and mean |SHAP|. Agreement across three independent views is stronger evidence than any single ranking.

## Models skipped

- `lightgbm`: missing package(s): lightgbm
- `xgboost`: missing package(s): xgboost
