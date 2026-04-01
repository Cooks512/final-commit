"""
Spread strategy for QMML Market Making Hackathon.
Calibrated from backtest results on training data.

Three tiers based on proven performance:

  ATTACK   (noise < 0.25)  — Go for MM. 85%+ coverage confirmed.
                              Stocks 1, 2.

  MODERATE (noise < 0.5)   — Competitive spread, comfortable as MM or trader.
                              Trader P&L +378/round in backtest.
                              Stock 3.

  SURVIVE  (noise >= 0.5)  — Wide spread, avoid MM. Trade minimum 10 shares.
                              Predictions are coin-flips; minimise bleeding.
                              Stocks 4–9.
"""

import numpy as np


def compute_spread(prediction, cv_rmse, profile, overfit_ratio=1.0,
                   aggression=None, capital=100_000, round_num=1):
    """
    Compute bid/ask with three-tier game-theory-aware spread sizing.
    """

    noise = profile.noise_ratio

    # ──────────────────────────────────────────────────────────
    # 1. DECIDE TIER
    # ──────────────────────────────────────────────────────────
    if aggression is not None:
        # Manual override — skip auto tier
        tier = "MANUAL"
    elif (noise < 0.25
          and overfit_ratio > 0.90
          and cv_rmse < prediction * 0.05):
        tier = "ATTACK"
    elif noise < 0.5:
        tier = "MODERATE"
    else:
        tier = "SURVIVE"

    # ──────────────────────────────────────────────────────────
    # 2. SET AGGRESSION PER TIER
    # ──────────────────────────────────────────────────────────
    if aggression is None:
        if tier == "ATTACK":
            aggression = 0.7       # tight — go for MM seat
        elif tier == "MODERATE":
            aggression = 0.4       # competitive but not reckless
        else:  # SURVIVE
            if noise < 0.8:
                aggression = 0.05  # wide
            else:
                aggression = 0.03  # very wide — just participating

    # ──────────────────────────────────────────────────────────
    # 3. COMPUTE HALF-SPREAD
    # ──────────────────────────────────────────────────────────
    multiplier  = 2.0 - 1.5 * aggression
    half_spread = cv_rmse * multiplier

    # Overfit penalty
    if overfit_ratio < 0.85:
        half_spread *= 1.0 + (0.85 - overfit_ratio)

    # OOD penalty
    half_spread *= profile.ood_spread_multiplier

    # ──────────────────────────────────────────────────────────
    # 4. BOUNDS PER TIER
    # ──────────────────────────────────────────────────────────
    if tier == "ATTACK":
        min_half = cv_rmse * 1.5
        max_half = cv_rmse * 3.0
    elif tier == "MODERATE":
        # Tighter than SURVIVE but still safe
        # Backtest showed 89.7% coverage at ~1x target_std spread
        min_half = cv_rmse * 1.2
        max_half = profile.target_std * 0.8
    else:  # SURVIVE
        min_half = profile.target_std * 0.5
        max_half = profile.target_std * 2.0

    half_spread = float(np.clip(half_spread, min_half, max_half))

    # ──────────────────────────────────────────────────────────
    # 5. MEAN-ANCHOR low-signal predictions
    # ──────────────────────────────────────────────────────────
    if noise > 0.8:
        mean_weight = min((noise - 0.8) / 0.2, 0.6)
        prediction = prediction * (1 - mean_weight) + profile.target_mean * mean_weight

    # ──────────────────────────────────────────────────────────
    # 6. ASYMMETRIC SPREAD (mild mean-reversion bias)
    # ──────────────────────────────────────────────────────────
    if profile.target_std > 0:
        deviation = (prediction - profile.target_mean) / profile.target_std
        asym = np.clip(deviation * 0.1, -0.2, 0.2)
    else:
        asym = 0

    bid_half = half_spread * (1 + asym)
    ask_half = half_spread * (1 - asym)

    bid    = round(prediction - bid_half, 2)
    ask    = round(prediction + ask_half, 2)
    spread = round(ask - bid, 2)

    return {
        "prediction":  round(prediction, 2),
        "bid":         bid,
        "ask":         ask,
        "spread":      spread,
        "aggression":  round(aggression, 3),
        "half_spread": round(half_spread, 2),
        "role":        tier,
    }


if __name__ == "__main__":
    from pathlib import Path
    import sys
    import pandas as pd
    sys.path.append(str(Path(__file__).parent))
    from data_check import profile_dataset
    from model_select import select_model

    DATA_DIR = Path(__file__).parent.parent / "hackathon_data"

    print(f"{'Stock':<6} {'Model':<28} {'pred':>7} {'bid':>7} {'ask':>7} "
          f"{'spread':>7} {'aggr':>6} {'noise':>6} {'ofit':>5} {'role':>10}")
    print("-" * 110)

    for i in range(1, 10):
        train = pd.read_csv(DATA_DIR / f"stock_{i}_train.csv")
        test  = pd.read_csv(DATA_DIR / f"stock_{i}_test.csv")
        X     = train.drop("target", axis=1)
        y     = train["target"]

        profile = profile_dataset(train, test)
        result  = select_model(profile, X, y, stock_number=i, verbose=False)

        train_pred    = result["model"].predict(X)
        train_rmse    = float(np.sqrt(np.mean((train_pred - y.values)**2)))
        overfit_ratio = train_rmse / result["cv_rmse"] if result["cv_rmse"] > 0 else 1.0

        pred = result["model"].predict(test)[0]

        quotes = compute_spread(
            pred, result["cv_rmse"], profile,
            overfit_ratio=overfit_ratio,
        )

        print(f"{i:<6} {result['name']:<28} {quotes['prediction']:>7.2f} "
              f"{quotes['bid']:>7.2f} {quotes['ask']:>7.2f} "
              f"{quotes['spread']:>7.2f} {quotes['aggression']:>6.3f} "
              f"{profile.noise_ratio:>6.2f} {overfit_ratio:>5.2f} "
              f"{quotes['role']:>10}")
