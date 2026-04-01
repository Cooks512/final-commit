"""
Spread strategy for the QMML market-making hackathon.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent))
from data_check import profile_dataset
from model_select import select_model


def compute_spread(prediction, cv_rmse, profile, overfit_ratio=1.0,
                   aggression=None, capital=100_000, round_num=1):
    """
    Compute bid/ask with tiered spread sizing.
    """

    if cv_rmse < 0:
        raise ValueError("cv_rmse must be non-negative.")

    noise = profile.noise_ratio

    if aggression is not None:
        aggression = float(np.clip(aggression, 0.0, 1.0))
        role = "MANUAL"
        if aggression >= 0.6:
            tier = "ATTACK"
        elif aggression >= 0.25:
            tier = "MODERATE"
        else:
            tier = "SURVIVE"
    elif (noise < 0.25 and overfit_ratio > 0.90 and cv_rmse < prediction * 0.05):
        tier = "ATTACK"
        role = tier
    elif noise < 0.5:
        tier = "MODERATE"
        role = tier
    else:
        tier = "SURVIVE"
        role = tier

    if aggression is None:
        if tier == "ATTACK":
            aggression = 0.7
        elif tier == "MODERATE":
            aggression = 0.4
        else:
            aggression = 0.05 if noise < 0.8 else 0.03

    multiplier = 2.0 - 1.5 * aggression
    half_spread = cv_rmse * multiplier

    if overfit_ratio < 0.85:
        half_spread *= 1.0 + (0.85 - overfit_ratio)

    half_spread *= profile.ood_spread_multiplier

    if tier == "ATTACK":
        min_half = cv_rmse * 1.5
        max_half = cv_rmse * 3.0
    elif tier == "MODERATE":
        min_half = cv_rmse * 1.2
        max_half = profile.target_std * 0.8
    else:
        min_half = profile.target_std * 0.5
        max_half = profile.target_std * 2.0

    half_spread = float(np.clip(half_spread, min_half, max_half))

    adjusted_prediction = float(prediction)
    if noise > 0.8:
        mean_weight = min((noise - 0.8) / 0.2, 0.6)
        adjusted_prediction = (
            adjusted_prediction * (1 - mean_weight) + profile.target_mean * mean_weight
        )

    if profile.target_std > 0:
        deviation = (adjusted_prediction - profile.target_mean) / profile.target_std
        asym = np.clip(deviation * 0.1, -0.2, 0.2)
    else:
        asym = 0

    bid_half = half_spread * (1 + asym)
    ask_half = half_spread * (1 - asym)

    bid = round(adjusted_prediction - bid_half, 2)
    ask = round(adjusted_prediction + ask_half, 2)
    spread = round(ask - bid, 2)

    return {
        "prediction": round(adjusted_prediction, 2),
        "prediction_raw": adjusted_prediction,
        "bid": bid,
        "ask": ask,
        "spread": spread,
        "aggression": round(aggression, 3),
        "half_spread": round(half_spread, 2),
        "role": role,
        "bounds_tier": tier,
    }


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "hackathon_data"

    print(
        f"{'Stock':<6} {'Model':<28} {'pred':>7} {'bid':>7} {'ask':>7} "
        f"{'spread':>7} {'aggr':>6} {'noise':>6} {'ofit':>5} {'role':>10}"
    )
    print("-" * 110)

    for i in range(1, 10):
        train = pd.read_csv(data_dir / f"stock_{i}_train.csv")
        test = pd.read_csv(data_dir / f"stock_{i}_test.csv")
        X = train.drop("target", axis=1)
        y = train["target"]

        profile = profile_dataset(train, test)
        result = select_model(profile, X, y, stock_number=i, verbose=False)

        train_pred = result["model"].predict(X)
        train_rmse = float(np.sqrt(np.mean((train_pred - y.values) ** 2)))
        overfit_ratio = train_rmse / result["cv_rmse"] if result["cv_rmse"] > 0 else 1.0

        pred = result["model"].predict(test)[0]
        quotes = compute_spread(
            pred,
            result["cv_rmse"],
            profile,
            overfit_ratio=overfit_ratio,
        )

        print(
            f"{i:<6} {result['name']:<28} {quotes['prediction']:>7.2f} "
            f"{quotes['bid']:>7.2f} {quotes['ask']:>7.2f} "
            f"{quotes['spread']:>7.2f} {quotes['aggression']:>6.3f} "
            f"{profile.noise_ratio:>6.2f} {overfit_ratio:>5.2f} "
            f"{quotes['role']:>10}"
        )
