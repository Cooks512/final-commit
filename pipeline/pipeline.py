"""
Helpers for loading and using saved stock-model artifacts.
"""

import pickle
from pathlib import Path

import pandas as pd


DEFAULT_SAVED_MODELS_DIR = Path(__file__).parent.parent / "saved_models"


def _resolve_model_path(model_path):
    path = Path(model_path)
    if path.exists():
        return path

    candidate = DEFAULT_SAVED_MODELS_DIR / path.name
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Saved model file not found: {model_path}")


def load_saved_model(model_path):
    """
    Load a saved stock-model artifact produced by `model_train.py`.

    Returns the full artifact dictionary, including:
    - `model`
    - `feature_columns`
    - `profile`
    - `model_summary`
    - `test_prediction`
    """
    resolved_path = _resolve_model_path(model_path)
    with open(resolved_path, "rb") as handle:
        artifact = pickle.load(handle)

    if isinstance(artifact, dict) and "model" in artifact:
        artifact.setdefault("artifact_path", str(resolved_path))
        return artifact

    if hasattr(artifact, "predict"):
        return {
            "model": artifact,
            "artifact_path": str(resolved_path),
            "feature_columns": None,
            "model_summary": {},
        }

    raise ValueError(f"Unsupported saved model format in {resolved_path}")


def deploy_model(saved_model, test_data):
    """
    Run prediction using either:
    - a full saved artifact returned by `load_saved_model`, or
    - a bare fitted model object.
    """
    artifact = saved_model if isinstance(saved_model, dict) else {"model": saved_model}
    model = artifact["model"]

    if isinstance(test_data, pd.DataFrame) and artifact.get("feature_columns"):
        feature_columns = artifact["feature_columns"]
        missing_columns = [column for column in feature_columns if column not in test_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required feature columns: {missing_columns}")
        model_input = test_data.loc[:, feature_columns].copy()
    else:
        model_input = test_data

    return model.predict(model_input)


def compute_spread(prediction):
    bid = 0
    ask = 0

    return {
        "bid": bid,
        "ask": ask,
    }

# Decide Role for round (Placeholder idk how ill input this yet into model)
def decide_role():
    return "Market" or "Trader"


# For a trader, we would want to decide whether to buy or sell based on the prediction and current market conditions.
def trader():
    return "Buy" or "Sell"


def one_round(number):
    model, test_data = load_saved_model(f"model_stock_{number}.pkl")
    prediction = deploy_model(model, test_data)
    spread = compute_spread(prediction)
    role, market_bid, market_ask = decide_role(prediction)
    if role == "Trader":
        trader(prediction, market_bid, market_ask)
    else:
        print(f"We are market makers with prediction {prediction} and spread {spread}")
        return 
