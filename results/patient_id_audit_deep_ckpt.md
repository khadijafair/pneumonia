# Patient id audit (deep features), original train vs test

Shared patient ids: 264

By label: NORMAL: 94, PNEUMONIA: 170

NORMAL shared ids by file-name prefix (train / test):

- train:IM / test:NORMAL2-IM: 94

## Image similarity (cosine of embeddings (outputs/densenet121/best.ckpt))

| label     | pairs                           |   n |   median_similarity |   mean |
|:----------|:--------------------------------|----:|--------------------:|-------:|
| NORMAL    | A same id train-test            |  94 |              0.8221 | 0.8012 |
| NORMAL    | B same id inside train          | 244 |              0.9081 | 0.8938 |
| NORMAL    | C different patients train-test |  94 |              0.8397 | 0.8172 |
| PNEUMONIA | A same id train-test            | 170 |              0.8061 | 0.8071 |
| PNEUMONIA | B same id inside train          | 500 |              0.8384 | 0.8252 |
| PNEUMONIA | C different patients train-test | 170 |              0.8312 | 0.8131 |

- NORMAL: sanity B > C, Mann-Whitney p = 1.2e-11 -> the measure separates people
- NORMAL: A > C for 38.3% of pairs, Mann-Whitney p = 8.8e-01
- PNEUMONIA: sanity B > C, Mann-Whitney p = 6.2e-02 -> INCONCLUSIVE measure
- PNEUMONIA: A > C for 45.3% of pairs, Mann-Whitney p = 9.6e-01

Reading: first B must be clearly above C (else the measure cannot tell people apart).
Then: A close to B and above C = shared ids are the same people (real leakage);
A close to C = the numbering is not shared between folders.
