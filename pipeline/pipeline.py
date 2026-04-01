"""
Helpers for loading saved stock models, quoting a spread, and making a trade
decision for a single round.
"""

import pickle
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

try:
    from .spread import compute_spread as _compute_quote_spread
    from .trading_decide import decide_trade as _decide_trade
except ImportError:
    from spread import compute_spread as _compute_quote_spread
    from trading_decide import decide_trade as _decide_trade


DEFAULT_DATA_DIR = Path(__file__).parent.parent / "hackathon_data"
DEFAULT_SAVED_MODELS_DIR = Path(__file__).parent.parent / "saved_models"


def _resolve_save_dir(save_dir=None):
    return Path(save_dir) if save_dir is not None else DEFAULT_SAVED_MODELS_DIR


def _resolve_data_dir(data_dir=None):
    return Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR


def _profile_object(profile_data):
    if isinstance(profile_data, dict):
        return SimpleNamespace(**profile_data)
    return profile_data


def load_saved_model(stock_number, save_dir=None, data_dir=None):
    """
    Load the saved model artifact and matching test data for one stock.
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


def predict_round(stock_number, save_dir=None, data_dir=None, aggression=None, capital=100_000, round_num=1):
    """
    Load one stock, predict its test price, and compute our quote.
    """
    loaded = load_saved_model(stock_number, save_dir=save_dir, data_dir=data_dir)
    prediction_values = deploy_model(loaded, loaded["test_data"])
    prediction = float(prediction_values[0])
    quote = compute_spread(
        loaded,
        prediction,
        aggression=aggression,
        capital=capital,
        round_num=round_num,
    )

    return {
        "stock_number": int(stock_number),
        "model_name": loaded.get("model_summary", {}).get("name"),
        "raw_prediction": prediction,
        "prediction": float(quote["prediction"]),
        "bid": float(quote["bid"]),
        "ask": float(quote["ask"]),
        "spread": float(quote["spread"]),
        "quote": {
            "raw_prediction": prediction,
            "prediction": float(quote["prediction"]),
            "prediction_raw": float(quote.get("prediction_raw", prediction)),
            "bid": float(quote["bid"]),
            "ask": float(quote["ask"]),
            "spread": float(quote["spread"]),
            "aggression": float(quote["aggression"]),
            "half_spread": float(quote["half_spread"]),
            "role": quote["role"],
            "bounds_tier": quote.get("bounds_tier", quote["role"]),
        },
        "loaded_model": loaded,
    }


def trade_round(
    loaded_model,
    prediction,
    mm_bid,
    mm_ask,
    capital,
    round_num=1,
    total_rounds=9,
    starting_capital=100_000,
    pnl_history=None,
):
    """
    Decide the trade to take against the market maker in one round.
    """
    profile = loaded_model.get("profile", {})
    model_summary = loaded_model.get("model_summary", {})

    decision = _decide_trade(
        prediction=float(prediction),
        cv_rmse=float(model_summary.get("cv_rmse", 0.0)),
        mm_bid=float(mm_bid),
        mm_ask=float(mm_ask),
        capital=float(capital),
        noise_ratio=float(model_summary.get("noise_ratio", profile.get("noise_ratio", 0.5))),
        target_mean=profile.get("target_mean"),
        target_std=profile.get("target_std"),
        round_num=round_num,
        total_rounds=total_rounds,
        starting_capital=starting_capital,
        pnl_history=pnl_history,
    )

    money_invested = float(decision["cost"])
    money_held = float(max(float(capital) - money_invested, 0.0))

    return {
        "action": decision["action"],
        "shares": int(decision["shares"]),
        "trade_price": float(decision["trade_price"]),
        "cost": float(decision["cost"]),
        "edge": float(decision["edge"]),
        "edge_zscore": float(decision["edge_zscore"]),
        "confidence": float(decision["confidence"]),
        "expected_pnl": float(decision["expected_pnl"]),
        "worst_case": float(decision["worst_case"]),
        "tier": decision["tier"],
        "trajectory": decision["trajectory"],
        "mm_confidence": decision["mm_confidence"],
        "running_sortino": decision["running_sortino"],
        "reasoning": decision["reasoning"],
        "affordability_limited": bool(decision.get("affordability_limited", False)),
        "market_bid": float(mm_bid),
        "market_ask": float(mm_ask),
        "market_mid": round((float(mm_bid) + float(mm_ask)) / 2.0, 2),
        "market_spread": round(float(mm_ask) - float(mm_bid), 2),
        "capital_before_trade": float(capital),
        "money_invested": round(money_invested, 2),
        "money_held": round(money_held, 2),
    }


def one_round(
    stock_number,
    mm_bid,
    mm_ask,
    capital,
    save_dir=None,
    data_dir=None,
    aggression=None,
    round_num=1,
    total_rounds=9,
    starting_capital=100_000,
    pnl_history=None,
):
    """
    Full round helper:
    - load saved model
    - predict fair value
    - compute our bid/ask spread
    - decide how to trade against the market maker
    """
    quote_result = predict_round(
        stock_number,
        save_dir=save_dir,
        data_dir=data_dir,
        aggression=aggression,
        capital=capital,
        round_num=round_num,
    )

    trade_result = trade_round(
        loaded_model=quote_result["loaded_model"],
        prediction=quote_result["prediction"],
        mm_bid=mm_bid,
        mm_ask=mm_ask,
        capital=capital,
        round_num=round_num,
        total_rounds=total_rounds,
        starting_capital=starting_capital,
        pnl_history=pnl_history,
    )

    return {
        "stock_number": int(stock_number),
        "model_name": quote_result["model_name"],
        "raw_prediction": quote_result["raw_prediction"],
        "prediction": quote_result["prediction"],
        "bid": quote_result["bid"],
        "ask": quote_result["ask"],
        "spread": quote_result["spread"],
        "quote": quote_result["quote"],
        "trade": trade_result,
        "model_summary": quote_result["loaded_model"].get("model_summary", {}),
        "profile": quote_result["loaded_model"].get("profile", {}),
    }
