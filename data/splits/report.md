# Data report

- Images: 5856 | patients: 2791 | unparsed file names: 1 | exact duplicates removed: 32

## Original Kaggle split

| orig_split   |   images |   patients |   pneumonia_rate |   bacteria |   virus |
|:-------------|---------:|-----------:|-----------------:|-----------:|--------:|
| train        |     5216 |       2636 |           0.7429 |       2530 |    1344 |
| val          |       16 |         15 |           0.5    |          8 |       0 |
| test         |      624 |        404 |           0.625  |        242 |     148 |

Leakage audit of the original split:

- patients_shared_train_val: 0
- images_in_val_of_patients_shared_train_val: 0
- identical_images_train_val: 0
- patients_shared_train_test: 264
- images_in_test_of_patients_shared_train_test: 416
- identical_images_train_test: 0
- patients_shared_val_test: 0
- images_in_test_of_patients_shared_val_test: 0
- identical_images_val_test: 0

## New patient-level split (seed 42)

| split   |   images |   patients |   pneumonia_rate |   bacteria |   virus |
|:--------|---------:|-----------:|-----------------:|-----------:|--------:|
| train   |     4160 |       1994 |           0.7291 |       1972 |    1060 |
| val     |      832 |        399 |           0.7284 |        389 |     217 |
| test    |      832 |        397 |           0.7284 |        399 |     207 |
