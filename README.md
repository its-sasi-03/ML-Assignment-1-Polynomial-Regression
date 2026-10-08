# ML Assignment 1 - Polynomial Regression

Roll number: BT2024124

## Models used

- var1: Polynomial Regression, degree 3, features x1, x2, x3
- var2: Polynomial Regression, degree 4, feature x1

These configurations follow the problem statement supplied for the assignment.

## Validation

5-fold cross-validation was used only on the training data.

- var1: MSE = 9.4454, R2 = 0.1595
- var2: MSE = 41.4010, R2 = 0.0969

## Running

Keep the four input CSV files and `train_and_predict.py` in the same folder, then run:

```bash
python train_and_predict.py
```

The script creates:
- BT2024124_pred_var1.csv
- BT2024124_pred_var2.csv
