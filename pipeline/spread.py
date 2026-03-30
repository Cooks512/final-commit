"""
Spread strategy for QMML Market Making Hackathon.

Game mechanics (from rules):
  - Tightest spread → you become the Market Maker
  - Market Maker MUST accept all trades from other teams
  - Other teams choose to buy at your ask or sell at your bid
  - True price revealed after → P&L calculated
  - Running out of money = elimination

Strategy implications:
  1. Being MM is HIGH RISK: every team trades against you with their own prediction.
     If your prediction is off, they all exploit the mispricing.
  2. You ONLY want to be MM when prediction confidence is genuinely high,
     AND your spread is wide enough to absorb the error.
  3. On low-confidence stocks: quote WIDE to avoid becoming MM.
     You'll profit by trading against whoever IS the MM using your prediction.
  4. Survival > any single round's profit. Sortino ratio rewards consistency.

Two modes:
  - ATTACK: go for MM on high-confidence stocks (noise_ratio < 0.25).
    Tight spread, but still covers ~1.5x CV RMSE.
  - SURVIVE: wide spread to avoid MM on everything else.
    Your edge comes from trading against the MM, not being one.
"""

import numpy as np


def compute_spread(prediction, cv_rmse, profile, overfit_ratio=1.0,
                   aggression=None, capital=100_000, round_num=1):
    """
    Compute bid/ask with game-theory-aware spread sizing.
    
    Args:
        prediction:    Model's point estimate for the stock price
        cv_rmse:       Cross-validated RMSE
        profile:       DataProfile from data_check
        overfit_ratio: train_rmse / cv_rmse (< 1 means overfitting)
        aggression:    Override aggression (None = auto from signal)
        capital:       Current capital (for position sizing context)
        round_num:     Which round (1-9) for survival weighting
    """

    noise = profile.noise_ratio

    # ──────────────────────────────────────────────────────────
    # 1. DECIDE ROLE: do we want to be Market Maker this round?
    # ──────────────────────────────────────────────────────────
    # Only attack when we have genuine edge AND low overfit risk
    want_mm = (
        noise < 0.25
        and overfit_ratio > 0.90
        and cv_rmse < prediction * 0.05   # RMSE < 5% of price
    )

    # ──────────────────────────────────────────────────────────
    # 2. SET AGGRESSION
    # ──────────────────────────────────────────────────────────
    if aggression is not None:
        pass  # manual override
    elif want_mm:
        # ATTACK mode: tight enough to likely win MM, wide enough to survive
        # Target: spread ≈ 3x CV RMSE (1.5x each side)
        aggression = 0.7
    else:
        # SURVIVE mode: we do NOT want to be MM
        # Quote wide — our profit comes from trading against the MM
        # Scale aggression down with noise: more noise → wider
        if noise < 0.5:
            aggression = 0.15       # moderately wide
        elif noise < 0.8:
            aggression = 0.08       # quite wide
        else:
            aggression = 0.03       # very wide — just participating

    # ──────────────────────────────────────────────────────────
    # 3. COMPUTE HALF-SPREAD
    # ──────────────────────────────────────────────────────────
    # Base multiplier: aggr=0.7 → 0.95x, aggr=0.15 → 1.775x, aggr=0.03 → 1.955x
    multiplier  = 2.0 - 1.5 * aggression
    half_spread = cv_rmse * multiplier

    # Overfit penalty
    if overfit_ratio < 0.85:
        half_spread *= 1.0 + (0.85 - overfit_ratio)

    # OOD penalty
    half_spread *= profile.ood_spread_multiplier

    # ──────────────────────────────────────────────────────────
    # 4. BOUNDS
    # ──────────────────────────────────────────────────────────
    if want_mm:
        # ATTACK: floor at 1.5x RMSE each side (must survive MM role)
        min_half = cv_rmse * 1.5
        max_half = cv_rmse * 3.0
    else:
        # SURVIVE: floor at 1x target_std, no real ceiling concern
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

    # Role label
    role = "ATTACK" if want_mm else "SURVIVE"

    return {
        "prediction":  round(prediction, 2),
        "bid":         bid,
        "ask":         ask,
        "spread":      spread,
        "aggression":  round(aggression, 3),
        "half_spread": round(half_spread, 2),
        "role":        role,
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
          f"{'spread':>7} {'aggr':>6} {'noise':>6} {'ofit':>5} {'role':>8}")
    print("-" * 105)

    for i in range(1, 10):
        train = pd.read_csv(DATA_DIR / f"stock_{i}_train.csv")
        test  = pd.read_csv(DATA_DIR / f"stock_{i}_test.csv")
        X     = train.drop("target", axis=1)
        y     = train["target"]

        profile = profile_dataset(train, test)
        result  = select_model(profile, X, y, verbose=False)

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
              f"{quotes['role']:>8}")