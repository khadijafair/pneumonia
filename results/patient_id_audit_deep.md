# Patient id audit (deep features), original train vs test

Shared patient ids: 264

By label: NORMAL: 94, PNEUMONIA: 170

NORMAL shared ids by file-name prefix (train / test):

- train:IM / test:NORMAL2-IM: 94

## Image similarity (cosine of embeddings (ImageNet DenseNet121))

| label     | pairs                           |   n |   median_similarity |   mean |
|:----------|:--------------------------------|----:|--------------------:|-------:|
| NORMAL    | A same id train-test            |  94 |              0.8806 | 0.8748 |
| NORMAL    | B same id inside train          | 244 |              0.8893 | 0.8837 |
| NORMAL    | C different patients train-test |  94 |              0.8803 | 0.8794 |
| PNEUMONIA | A same id train-test            | 170 |              0.8491 | 0.8472 |
| PNEUMONIA | B same id inside train          | 500 |              0.8616 | 0.8576 |
| PNEUMONIA | C different patients train-test | 170 |              0.8551 | 0.8506 |

- NORMAL: sanity B > C, Mann-Whitney p = 2.9e-02 -> INCONCLUSIVE measure
- NORMAL: A > C for 43.6% of pairs, Mann-Whitney p = 6.9e-01
- PNEUMONIA: sanity B > C, Mann-Whitney p = 2.4e-02 -> INCONCLUSIVE measure
- PNEUMONIA: A > C for 49.4% of pairs, Mann-Whitney p = 9.0e-01

Reading: first B must be clearly above C (else the measure cannot tell people apart).
Then: A close to B and above C = shared ids are the same people (real leakage);
A close to C = the numbering is not shared between folders.
