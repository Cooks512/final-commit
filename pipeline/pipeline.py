"""
Helpers for loading and using saved stock-model artifacts.
"""

import pickle
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

try:
    from .spread import compute_spread as _compute_quote_spread
except ImportError:
    from spread import compute_spread as _compute_quote_spread


DEFAULT_DATA_DIR = Path(__file__).parent.parent / "hackathon_data"
DEFAULT_SAVED_MODELS_DIR = Path(__file__).parent.parent / "saved_models"


def _resolve_save_dir(save_dir=None):
    return Path(save_dir) if save_dir is not None else DEFAULT_SAVED_MODELS_DIR


def _resolve_data_dir(data_dir=None):
    return Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR


def load_saved_model(stock_number, save_dir=None, data_dir=None):
    """
    Load the saved model artifact and matching test data for one stock.

    Usage:
        loaded = load_saved_model(3)
    """
    stock_number = int(stock_number)
    save_dir = _resolve_save_dir(save_dir)
    data_dir = _resolve_data_dir(data_dir)

    model_path = save_dir / f"stock_{stock_number}_winner.pkl"
    test_path = data_dir / f"stock_{stock_number}_test.csv"

    if not model_path.exists():
        raise FileNotFoundError(f"Saved model file not found: {model_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test data file not found: {test_path}")

    with open(model_path, "rb") as handle:
        artifact = pickle.load(handle)

    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError(f"Unsupported saved model format in {model_path}")

    return {
        "stock_number": artifact.get("stock_number", stock_number),
        "profile": artifact.get("profile", {}),
        "model": artifact["model"],
        "model_summary": artifact.get("model_summary", {}),
        "test_prediction": artifact.get("test_prediction", []),
        "feature_columns": artifact.get("feature_columns"),
        "test_data": pd.read_csv(test_path),
    }


def deploy_model(saved_model, test_data):
    """
    Run prediction using a loaded model artifact or a bare fitted model.
    """
    loaded = saved_model if isinstance(saved_model, dict) else {"model": saved_model}
    model = loaded["model"]

    if isinstance(test_data, pd.DataFrame) and loaded.get("feature_columns"):
        feature_columns = loaded["feature_columns"]
        missing_columns = [column for column in feature_columns if column not in test_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required feature columns: {missing_columns}")
        model_input = test_data.loc[:, feature_columns].copy()
    else:
        model_input = test_data

    return model.predict(model_input)


def _profile_object(profile_data):
    if isinstance(profile_data, dict):
        return SimpleNamespace(**profile_data)
    return profile_data


def compute_spread(loaded_model, prediction, aggression=None, capital=100_000, round_num=1):
    """
    Compute bid/ask using the saved model's profile and model summary.
    """
    model_summary = loaded_model.get("model_summary", {})
    profile = _profile_object(loaded_model.get("profile", {}))

    cv_rmse = model_summary.get("cv_rmse")
    if cv_rmse is None:
        raise ValueError("Saved model is missing `cv_rmse`, which is required for spread calculation.")

    return _compute_quote_spread(
        prediction=float(prediction),
        cv_rmse=float(cv_rmse),
        profile=profile,
        overfit_ratio=float(model_summary.get("overfit_ratio", 1.0)),
        aggression=aggression,
        capital=capital,
        round_num=round_num,
    )

def decide_role():
    return "Market" or "Trader"


def trader():
    shares_to_trade = 0
    num = max(10, shares_to_trade)
    return "Buy" or "Sell", num


def predict_round(number):
    loaded = load_saved_model(number)
    prediction_values = deploy_model(loaded, loaded["test_data"])
    prediction = float(prediction_values[0])
    quotes = compute_spread(loaded, prediction)

    return {
        "prediction": float(quotes["prediction"]),
        "bid": float(quotes["bid"]),
        "ask": float(quotes["ask"]),
        "spread": float(quotes["spread"]),
    }

def trade_round(predictions, market_ask, market_bid, money):
    pass

def main():
    money = 100_000
    for i in range(1, 10):
        predictions = predict_round(i)
        role = decide_role()
        if role == "Trader":
            #HOW DO I GET MARKET ASK AND BID HERE?
            market_ask = 0
            market_bid = 0

            money = trade_round(predictions, market_ask, market_bid, money)
        else:
            pass

    return money
        

stuff = predict_round(3)
print(stuff)