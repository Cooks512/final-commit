import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold, KFold

SEED = 2003

#Return cross-validation strategy based on the number of rows in the training data
def cross_validate(n_rows, seed=SEED, heavy_but_betterSlighly=False):
    
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
def train():
    pass

def main():
    train_data_1, test_data_1 = load_data(1)
    explore_data(train_data_1, test_data_1)
    x_train_1, y_train_1, x_test_1, baseline_rmse_1 = prepare_data(train_data_1, test_data_1)