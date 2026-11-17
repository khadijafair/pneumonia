# Patient id audit (train vs test of the original Kaggle split)

Shared patient ids: 264

By label: NORMAL: 94, PNEUMONIA: 170

NORMAL shared ids by file-name prefix (train / test):

- train:IM / test:NORMAL2-IM: 94

## Image similarity (Pearson correlation, 128x128 grayscale)

| label     | pairs                           |   n |   median_similarity |   mean |
|:----------|:--------------------------------|----:|--------------------:|-------:|
| NORMAL    | A same id train-test            |  94 |               0.711 |  0.705 |
| NORMAL    | B same id inside train          | 244 |               0.73  |  0.72  |
| NORMAL    | C different patients train-test |  94 |               0.725 |  0.705 |
| PNEUMONIA | A same id train-test            | 170 |               0.716 |  0.691 |
| PNEUMONIA | B same id inside train          | 500 |               0.725 |  0.704 |
| PNEUMONIA | C different patients train-test | 170 |               0.71  |  0.689 |

- NORMAL: A > C for 48.9% of pairs, Mann-Whitney p = 6.1e-01
- PNEUMONIA: A > C for 50.0% of pairs, Mann-Whitney p = 5.0e-01

Reading: if A is close to B and clearly above C, the shared ids are the same people
(real leakage). If A is close to C, the numbering is not shared between folders.
