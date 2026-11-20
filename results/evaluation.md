# Step 3 — Clinical evaluation

Test set: 832 images, 397 patients. Thresholds chosen on validation only. 95 % CI: patient bootstrap, n = 2000.

## Threshold-free metrics (test)

| model | AUROC [95 % CI] | AUPRC | Brier | ECE |
|---|---|---|---|---|
| densenet121 | 0.9964 [0.9937 ; 0.9986] | 0.9987 | 0.0205 | 0.0296 |
| efficientnet_b0 | 0.9961 [0.9927 ; 0.9987] | 0.9985 | 0.0367 | 0.0448 |
| vit_small | 0.9966 [0.9938 ; 0.9989] | 0.9988 | 0.0249 | 0.0357 |

## Paired AUROC differences (test, same bootstrap patients)

- densenet121 - efficientnet_b0: +0.0003 [-0.0015 ; +0.0023] → not significant (CI contains 0)
- densenet121 - vit_small: -0.0002 [-0.0021 ; +0.0021] → not significant (CI contains 0)
- efficientnet_b0 - vit_small: -0.0005 [-0.0030 ; +0.0015] → not significant (CI contains 0)

## Operating point: threshold set on validation for sensitivity ≥ 98%, applied to test

| model | threshold | sensitivity [95 % CI] | specificity [95 % CI] | PPV | NPV | missed pneumonias (FN) | false alarms (FP) |
|---|---|---|---|---|---|---|---|
| densenet121 | 0.6094 | 97.9 % [96.6 % ; 99.0 %] | 96.5 % [94.0 % ; 98.7 %] | 98.7 % | 94.4 % | 13 | 8 |
| efficientnet_b0 | 0.0590 | 98.3 % [97.3 % ; 99.3 %] | 97.8 % [95.7 % ; 99.6 %] | 99.2 % | 95.7 % | 10 | 5 |
| vit_small | 0.1913 | 98.2 % [97.1 % ; 99.2 %] | 96.5 % [93.9 % ; 98.6 %] | 98.7 % | 95.2 % | 11 | 8 |

## For reference: Youden threshold (val) and default 0.5

| model | Youden threshold | sensitivity | specificity | sens. @0.5 | spec. @0.5 |
|---|---|---|---|---|---|
| densenet121 | 0.7378 | 96.2 % | 98.7 % | 98.3 % | 94.2 % |
| efficientnet_b0 | 0.1312 | 97.0 % | 97.8 % | 93.7 % | 98.7 % |
| vit_small | 0.3564 | 97.2 % | 97.8 % | 95.7 % | 98.7 % |

## Sensitivity by pneumonia subtype (test, operating threshold)

| model | bacterial | viral |
|---|---|---|
| densenet121 | 99.0 % (n = 399) | 95.7 % (n = 207) |
| efficientnet_b0 | 99.0 % (n = 399) | 97.1 % (n = 207) |
| vit_small | 99.2 % (n = 399) | 96.1 % (n = 207) |

Errors per model: `errors_<model>.csv` (to inspect visually).

![ROC and calibration](roc_calibration.png)
