# Off-Spec Risk Model — Evaluation Report

**Selected model:** LightGBM  
**Selection criterion:** highest validation pr_auc, tie-broken on median_warning_min  
**Operating threshold:** 0.6000 (chosen on validation via `f1`)  
**Prediction target:** basis weight deviates more than 2.5% from setpoint within the next 10 minutes

## Leakage controls

- Splits are **event-wise**, never row-wise.
- Features are strictly backward-looking (asserted by `test_features.py::test_no_future_leakage`).
- Threshold selected on validation only.
- Model selection used validation only.
- **Test split scored exactly once**, after the winner was fixed.
- Permutation importance computed on validation, not training.

## Operating point

Alarm on-delay: **3 samples (15 s)** -- an alarm is confirmed only after the risk score stays above threshold for that long (ISA-18.2 on-delay).

The threshold and on-delay are tuned on an **event-level** objective, not row-level F1. A transition is ~336 samples long, so even a 6% row-level false positive rate makes a nuisance alarm near-certain in every clean transition, and a system that does that gets switched off. Conversely row-level recall understates usefulness: an excursion only has to be caught once to warn the operator. The objective is therefore to minimise per-clean-event false alarms subject to a floor on event detection rate.

## Evaluation population

Classification metrics are computed on rows where basis weight is still **inside** the ±2.5% band. Predicting a breach while the sheet is already off-spec is trivial and would inflate every score. Early-warning metrics necessarily span all rows, since they measure the gap to the breach itself.

## Split composition

| split   |   events |   rows |   positive_rate |   in_spec_rows |   in_spec_positive_rate |
|:--------|---------:|-------:|----------------:|---------------:|------------------------:|
| train   |      304 | 102144 |        0.289591 |          91943 |                0.212806 |
| val     |       98 |  32928 |        0.298287 |          29950 |                0.230551 |
| test    |       98 |  32928 |        0.308522 |          29157 |                0.221148 |

## Model comparison (validation)

| model                  |   pr_auc |   precision |   recall |       f1 |       fpr |   median_warning_min |   mean_warning_min |   detection_rate |   false_alarm_event_rate |
|:-----------------------|---------:|------------:|---------:|---------:|----------:|---------------------:|-------------------:|-----------------:|-------------------------:|
| xgboost                | 0.781757 |    0.823733 | 0.517741 | 0.635838 | 0.0331959 |              4.58333 |            4.89966 |         0.830508 |                 0.205128 |
| lightgbm               | 0.775609 |    0.814715 | 0.52281  | 0.636909 | 0.0356259 |              4.41667 |            4.96599 |         0.830508 |                 0.128205 |
| hist_gradient_boosting | 0.771776 |    0.823253 | 0.506589 | 0.627219 | 0.0325884 |              4.58333 |            4.77    |         0.847458 |                 0.179487 |
| random_forest          | 0.761306 |    0.846173 | 0.499493 | 0.628176 | 0.0272076 |              4.41667 |            4.80952 |         0.830508 |                 0.230769 |
| logistic_regression    | 0.745591 |    0.743802 | 0.560463 | 0.639247 | 0.0578433 |              4.91667 |            5.71007 |         0.813559 |                 0.333333 |

## Selected model — validation

| Metric | Value |
|---|---|
| PR-AUC | 0.7756 |
| Precision | 0.8147 |
| Recall | 0.5228 |
| F1 | 0.6369 |
| False positive rate | 0.0356 |
| Median warning time | 4.42 min |
| Mean warning time | 4.97 min |
| **Event detection rate** | 0.831 |
| False alarm rate (per clean event) | 0.128 |

## Selected model — test (held out, scored once)

| Metric | Value |
|---|---|
| PR-AUC | 0.8208 |
| Precision | 0.8016 |
| Recall | 0.5802 |
| F1 | 0.6731 |
| False positive rate | 0.0408 |
| Median warning time | 4.50 min |
| Mean warning time | 5.28 min |
| **Event detection rate** | 0.847 |
| False alarm rate (per clean event) | 0.179 |

## Explainability

SHAP method: `shap.TreeExplainer` (exact=True), 600 rows explained.

### Top 15 features by consensus attribution

| feature                         |   native_norm |   permutation_norm |   shap_norm |   consensus |
|:--------------------------------|--------------:|-------------------:|------------:|------------:|
| t_since_ramp_min                |    0.033156   |        0.249335    |   0.123591  |   0.135361  |
| bw_dev_headroom_pct             |    0.00925532 |        0.100752    |   0.0649222 |   0.05831   |
| plan_ramp_min                   |    0.052234   |        0.0667002   |   0.0289877 |   0.0493073 |
| plan_trim_enabled               |    0.0102128  |        0.103716    |   0.0301224 |   0.048017  |
| bw_abs_dev_pct                  |    0.00716312 |        0.0582757   |   0.0509456 |   0.0387948 |
| ramp_progress                   |    0.00925532 |        0.0513355   |   0.0507458 |   0.0371122 |
| plan_lead_scale                 |    0.0518794  |        0.0171677   |   0.0259236 |   0.0316569 |
| plan_tau_c_scale                |    0.0517376  |        0.00603582  |   0.0356603 |   0.0311446 |
| transition_magnitude            |    0.0159574  |        0.0378509   |   0.0307652 |   0.0281912 |
| mv_machine_speed_slew_util      |    0.0158156  |        0.0245632   |   0.0341    |   0.0248263 |
| mv_stock_flow_remaining         |    0.021383   |        0.0167263   |   0.0298295 |   0.0226463 |
| dv_headbox_consistency_roc_2min |    0.0150355  |        0.0335917   |   0.012981  |   0.020536  |
| mv_machine_speed_remaining_frac |    0.012766   |        0.0114243   |   0.0253905 |   0.0165269 |
| mv_stock_flow_remaining_frac    |    0.0167021  |        0.00825725  |   0.0229227 |   0.0159607 |
| ramp_deficit_min                |    0.0135106  |        0.000738207 |   0.0269913 |   0.0137467 |

Consensus averages the model's native importance, permutation importance on validation, and mean |SHAP|. Agreement across three independent views is stronger evidence than any single ranking.
