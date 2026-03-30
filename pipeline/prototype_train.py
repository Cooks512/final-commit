import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold, KFold, cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

SEED = 2003

#Return cross-validation strategy based on the number of rows in the training data
def cross_validate(n_rows, seed=SEED, heavy_but_better=False):
    
    if heavy_but_better:
        return RepeatedKFold(n_splits=5, n_repeats=10, random_state=seed)
    
    if n_rows < 60:
        return RepeatedKFold(n_splits=5, n_repeats=5, random_state=seed)
    elif n_rows < 200:
        return RepeatedKFold(n_splits=5, n_repeats=3, random_state=seed)
    else:
        return KFold(n_splits=5, shuffle=True, random_state=seed)

#Function to explore a given stock's training and testing data
def explore_data(train_data, test_data):
    x_train = train_data.drop("target", axis=1)
    y_train = train_data["target"]

    train_shape = x_train.shape
    test_shape = test_data.shape

    print(f"Training data has {train_shape[0]} rows and {train_shape[1]} features.")
    print(f"Testing data has {test_shape[0]} rows and {test_shape[1]} features.")

    print(f"Averge target stock price: {y_train.mean()}")
    print(f"Spread of target stock price: {y_train.std()}")  # Baseline RMSE

    print(f"Correlation of features with target: {train_data.corr(numeric_only=True)['target']}")

    correlations = train_data.corr(numeric_only=True)["target"].drop("target")
    print("Features sorted by absolute correlation with target:")
    print(correlations.abs().sort_values(ascending=False))

    if (train_data.isnull().sum().any()):
        print("Missing values found in training data:")
        print(train_data.isnull().sum())
    else:
        print("No missing values found in training data.")

#Split the training data into features and target, scale the features, and calculate baseline RMSE
def prepare_data(train_data, test_data):
    x_train = train_data.drop("target", axis=1)
    y_train = train_data["target"]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(test_data)

    baseline_rmse = y_train.std()

    return x_train_scaled, y_train, x_test_scaled, baseline_rmse

#Create paths for train and test data based on stock number
def get_data_path(stock_number, train_test="train"):
    if train_test == "train":
        return f"data/stock_{stock_number}_train.csv"
    elif train_test == "test":
        return f"data/stock_{stock_number}_test.csv"
    else:
        raise ValueError("train_test must be 'train' or 'test'")

#Load  train and test data for a given stock number
def load_data(stock_number):
    train_path = get_data_path(stock_number, "train")
    test_path = get_data_path(stock_number, "test")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Testing data not found at {test_path}")

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    return train_data, test_data

#Train model
def train_ridge(x_scaled, y_train, x_test_scaled, baseline_rmse, cv):
    best_rmse = baseline_rmse
    best_predictions = y_train.mean()
    best_alpha = None

    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        model = Ridge(alpha=alpha)
        scores = cross_val_score(model, x_scaled, y_train, cv=cv, scoring="neg_root_mean_squared_error")
        rmse = -scores.mean()

        model.fit(x_scaled, y_train)
        predictions = model.predict(x_test_scaled)[0]

        print(f"Alpha: {alpha}, CV RMSE: {rmse:.4f}, Test Prediction: {predictions:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_predictions = predictions
            best_alpha = alpha

    return best_rmse, best_predictions, best_alpha

def train_gradient_boost(x_scaled, y_train, x_test_scaled, baseline_rmse, cv, seed=SEED):
    best_rmse = baseline_rmse
    best_predictions = y_train.mean()
    best_params = None

    configs = [
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05},
        {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05},
    ]

    for params in configs:
        model = GradientBoostingRegressor(**params, random_state=seed)
        scores = cross_val_score(model, x_scaled, y_train, cv=cv, scoring="neg_root_mean_squared_error")
        rmse = -scores.mean()

        model.fit(x_scaled, y_train)
        predictions = model.predict(x_test_scaled)[0]

        print(f"Config: {params}, CV RMSE: {rmse:.4f}, Test Prediction: {predictions:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_predictions = predictions
            best_params = params

    return best_rmse, best_predictions, best_params
    
def analyze_stock(stock_number):
    train_data, test_data = load_data(stock_number)
    explore_data(train_data, test_data)

    x_train, y_train, x_test, baseline_rmse = prepare_data(train_data, test_data)
    cv = cross_validate(len(y_train))

    #Fit data on diffrent models 
    ridge_rmse, ridge_prediction, ridge_alpha = train_ridge(x_train, y_train, x_test, baseline_rmse, cv)
    gb_rmse, gb_prediction, gb_params = train_gradient_boost(x_train, y_train, x_test, baseline_rmse, cv)

    if ridge_rmse <= gb_rmse:
        best_rmse = ridge_rmse
        best_prediction = ridge_prediction
        best_model = f"Ridge(alpha={ridge_alpha})"
    else:
        best_rmse = gb_rmse
        best_prediction = gb_prediction
        best_model = f"GradientBoostingRegressor({gb_params})" 

    return {
        "stock_number": stock_number,
        "best_rmse": best_rmse,
        "baseline_rmse": baseline_rmse,
        "best_prediction": best_prediction,
        "best_model": best_model,
    }

def main():
    results = []

    for stock_number in range(1, 10):
        print(f"\n--- Stock {stock_number} ---")
        results.append(analyze_stock(stock_number))

    print("\nSummary")
    for result in results:
        print(
            f"Best RMSE for Stock {result['stock_number']}: "
            f"{result['best_rmse']:.4f} with model {result['best_model']}, "
            f"Test Prediction: {result['best_prediction']:.4f}"
        )

if __name__ == "__main__":
    main()
