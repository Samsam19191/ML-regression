import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
import joblib

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
pd.set_option("display.max_columns", None)

### PART 1 : preprocessing of training data :
# 1. remove the first column (id)
# 2. fill missing values or drop rows with missing values depending on the amount of missing data
# 3. convert useful text columns to numerical values using one-hot encoding or label encoding

# drop 'hc' column (too many missing values)
# train_df = train_df.drop(columns=["id", "hc", "model", "range"])
# test_df = test_df.drop(columns=["id", "hc", "model", "range"])

# try dropping all non-numerical columns (+ hc)
train_df = train_df.drop(
    columns=[
        "id",
        "brand",
        "model",
        "car_class",
        "range",
        "fuel_type",
        "hybrid",
        "grbx_type_ratios",
        "hc",
    ]
)

# drop rows with missing urb_cons and exturb_cons values

# train_df = train_df.dropna(subset=["urb_cons", "exturb_cons"])
# test_df = test_df.dropna(subset=["urb_cons", "exturb_cons"])

# try dropping all rows with missing valued

train_df = train_df.dropna(
    subset=["urb_cons", "exturb_cons", "nox", "co", "hcnox", "ptcl"]
)
test_df = test_df.dropna(
    subset=["urb_cons", "exturb_cons", "nox", "co", "hcnox", "ptcl"]
)

# # for nox, co, hcnox, ptcl, fill missing values with the mean of the column

# train_df["nox"] = train_df["nox"].fillna(train_df["nox"].mean())
# test_df["nox"] = test_df["nox"].fillna(test_df["nox"].mean())

# train_df["co"] = train_df["co"].fillna(train_df["co"].mean())
# test_df["co"] = test_df["co"].fillna(test_df["co"].mean())

# train_df["hcnox"] = train_df["hcnox"].fillna(train_df["hcnox"].mean())
# test_df["hcnox"] = test_df["hcnox"].fillna(test_df["hcnox"].mean())

# train_df["ptcl"] = train_df["ptcl"].fillna(train_df["ptcl"].mean())
# test_df["ptcl"] = test_df["ptcl"].fillna(test_df["ptcl"].mean())

# # convert brand, car_class to numerical values using one-hot encoding
# train_df = pd.get_dummies(
#     train_df, columns=["brand", "car_class", "fuel_type", "grbx_type_ratios"]
# )
# test_df = pd.get_dummies(
#     test_df, columns=["brand", "car_class", "fuel_type", "grbx_type_ratios"]
# )

# verify what valued the hybrid column can take
# print(df["hybrid"].value_counts())

# # convert hybrid column to binary values
# train_df["hybrid"] = train_df["hybrid"].map({"non": 0, "oui": 1})
# test_df["hybrid"] = test_df["hybrid"].map({"non": 0, "oui": 1})

# Split data
X = train_df.drop(columns=["co2"])
y = train_df["co2"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

### PART 2 : training a regression model
## basic model first

# # Train using diffrent models
# # model = RandomForestRegressor(n_estimators=100, random_state=42)
# # model = XGBRegressor(n_estimators=100, learning_rate=0.1)
# model = DecisionTreeRegressor(random_state=42)
# model.fit(X_train, y_train)

# # Evaluate
# y_pred = model.predict(X_test)
# mae = mean_absolute_error(y_test, y_pred)
# print(f"Mean Absolute Error: {mae}")
# relative_error = (mae / train_df["co2"].mean()) * 100
# print(f"Relative Error: {relative_error:.2f}%")

## tuning hyperparameters

# param_grid = {
#     "max_depth": [5, 10, 15, None],
#     "min_samples_split": [2, 5, 10],
#     "min_samples_leaf": [1, 2, 5],
# }

# grid_search = GridSearchCV(
#     DecisionTreeRegressor(random_state=42),
#     param_grid,
#     cv=5,
#     scoring="neg_mean_absolute_error",
# )
# grid_search.fit(X_train, y_train)

# best_model = grid_search.best_estimator_
# print("Best hyperparameters:", grid_search.best_params_)

# # Save the model with the best parameters
# joblib.dump(best_model, "best_model.pkl")

# train a model with the best params
best_model = joblib.load("best_model.pkl")
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
relative_error = (mae / train_df["co2"].mean()) * 100
print(f"Relative Error: {relative_error:.2f}%")
