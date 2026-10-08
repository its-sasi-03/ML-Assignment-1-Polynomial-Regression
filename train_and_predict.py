import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

ROLLNO = "BT2024124"

def make_model(degree):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("linear", LinearRegression())
    ])

# var1: degree 3, first 3 features
train1 = pd.read_csv(f"{ROLLNO}_train_var1.csv")
test1 = pd.read_csv(f"{ROLLNO}_test_var1.csv")

features1 = ["x1", "x2", "x3"]
model1 = make_model(3)
model1.fit(train1[features1], train1["y"])

pred1 = model1.predict(test1[features1])
pd.DataFrame({"y": pred1}).to_csv(f"{ROLLNO}_pred_var1.csv", index=False)

# var2: degree 4, first feature
train2 = pd.read_csv(f"{ROLLNO}_train_var2.csv")
test2 = pd.read_csv(f"{ROLLNO}_test_var2.csv")

features2 = ["x1"]
model2 = make_model(4)
model2.fit(train2[features2], train2["y"])

pred2 = model2.predict(test2[features2])
pd.DataFrame({"y": pred2}).to_csv(f"{ROLLNO}_pred_var2.csv", index=False)

print("Prediction files created.")
