"""
Trade decision helper for the QMML market-making hackathon.

The goal is to stay defensive by default:
- use model edge and confidence when it is real
- respect capital and affordability limits
- fall back to minimum-size trades when signal is weak
"""

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent))
from data_check import profile_dataset
from model_select import select_model


def _validate_trade_inputs(mm_bid, mm_ask, capital, min_shares, max_risk_pct, cv_rmse):
    if mm_bid < 0 or mm_ask <= 0:
        raise ValueError("Market-maker prices must be positive, with bid >= 0 and ask > 0.")
    if mm_ask <= mm_bid:
        raise ValueError("Market-maker ask must be greater than bid.")
    if capital < 0:
        raise ValueError("Capital must be non-negative.")
    if min_shares <= 0:
        raise ValueError("min_shares must be positive.")
    if not (0 < max_risk_pct <= 1):
        raise ValueError("max_risk_pct must be in the interval (0, 1].")
    if cv_rmse < 0:
        raise ValueError("cv_rmse must be non-negative.")


def _determine_tier(noise_ratio):
    if noise_ratio < 0.25:
        return "ATTACK"
    if noise_ratio < 0.5:
        return "MODERATE"
    return "SURVIVE"


def _capital_trajectory(capital, starting_capital, round_num, total_rounds, max_risk_pct):
    capital_ratio = capital / starting_capital if starting_capital > 0 else 1.0
    rounds_left = total_rounds - round_num + 1

    if capital_ratio > 1.15:
        trajectory = "AHEAD"
        traj_mult = 1.3
    elif capital_ratio > 1.0:
        trajectory = "SLIGHT_UP"
        traj_mult = 1.1
    elif capital_ratio > 0.85:
        trajectory = "SLIGHT_DOWN"
        traj_mult = 0.8
    else:
        trajectory = "DANGER"
        traj_mult = 0.4
        max_risk_pct = 0.05

    if rounds_left <= 3:
        if trajectory == "AHEAD":
            traj_mult *= 0.7
        elif trajectory == "DANGER":
            traj_mult *= 1.5

    return capital_ratio, trajectory, traj_mult, max_risk_pct


def _market_maker_confidence(mm_spread, target_std):
    mm_confidence = "UNKNOWN"
    mm_mult = 1.0

    if target_std and target_std > 0:
        spread_ratio = mm_spread / target_std
        if spread_ratio < 0.3:
            mm_confidence = "VERY_CONFIDENT"
            mm_mult = 0.5
        elif spread_ratio < 0.6:
            mm_confidence = "CONFIDENT"
            mm_mult = 0.8
        elif spread_ratio < 1.0:
            mm_confidence = "UNCERTAIN"
            mm_mult = 1.0
        else:
            mm_confidence = "VERY_UNCERTAIN"
            mm_mult = 1.3

    return mm_confidence, mm_mult


def _share_limits(capital, trade_price, min_shares, max_risk_pct):
    max_risk = capital * max_risk_pct
    max_shares_by_risk = int(max_risk / trade_price) if trade_price > 0 else 0
    max_affordable_shares = int(capital / trade_price) if trade_price > 0 else 0
    effective_min_shares = min(min_shares, max_affordable_shares)
    size_ceiling = max(
        effective_min_shares,
        min(max_shares_by_risk, max_affordable_shares),
    )
    affordability_limited = max_affordable_shares < min_shares
    return {
        "max_shares_by_risk": max_shares_by_risk,
        "max_affordable_shares": max_affordable_shares,
        "effective_min_shares": effective_min_shares,
        "size_ceiling": size_ceiling,
        "affordability_limited": affordability_limited,
    }


def decide_trade(prediction, cv_rmse, mm_bid, mm_ask, capital,
                 noise_ratio=0.5, target_mean=None, target_std=None,
                 min_shares=10, max_risk_pct=0.15,
                 round_num=1, total_rounds=9, starting_capital=100_000,
                 pnl_history=None):
    """
    Decide whether to buy or sell against the market maker.
    """

    if pnl_history is None:
        pnl_history = []

    _validate_trade_inputs(
        mm_bid=mm_bid,
        mm_ask=mm_ask,
        capital=capital,
        min_shares=min_shares,
        max_risk_pct=max_risk_pct,
        cv_rmse=cv_rmse,
    )

    mm_mid = (mm_bid + mm_ask) / 2
    mm_spread = mm_ask - mm_bid
    tier = _determine_tier(noise_ratio)

    capital_ratio, trajectory, traj_mult, max_risk_pct = _capital_trajectory(
        capital=capital,
        starting_capital=starting_capital,
        round_num=round_num,
        total_rounds=total_rounds,
        max_risk_pct=max_risk_pct,
    )
    mm_confidence, mm_mult = _market_maker_confidence(mm_spread, target_std)

    buy_edge = prediction - mm_ask
    sell_edge = mm_bid - prediction

    if buy_edge > 0:
        action = "BUY"
        edge = buy_edge
        trade_price = mm_ask
    elif sell_edge > 0:
        action = "SELL"
        edge = sell_edge
        trade_price = mm_bid
    else:
        edge = 0.0
        if tier == "SURVIVE" and target_mean is not None:
            if mm_mid > target_mean:
                action = "SELL"
                trade_price = mm_bid
            else:
                action = "BUY"
                trade_price = mm_ask
        elif prediction >= mm_mid:
            action = "BUY"
            trade_price = mm_ask
        else:
            action = "SELL"
            trade_price = mm_bid

    share_limits = _share_limits(
        capital=capital,
        trade_price=trade_price,
        min_shares=min_shares,
        max_risk_pct=max_risk_pct,
    )
    effective_min_shares = share_limits["effective_min_shares"]
    size_ceiling = share_limits["size_ceiling"]
    affordability_limited = share_limits["affordability_limited"]

    edge_zscore = edge / cv_rmse if cv_rmse > 0 else 0.0
    confidence = max(0.05, 1.0 - noise_ratio)

    if tier == "SURVIVE":
        shares = effective_min_shares
        reasoning = f"SURVIVE (noise={noise_ratio:.2f}) - minimum {shares} shares."
        if edge == 0 and target_mean is not None:
            direction = "above" if action == "SELL" else "below"
            reasoning += f" Mean-reversion: MM mid {direction} target mean."
    else:
        if edge_zscore <= 0:
            base_frac = 0.0
        else:
            base_frac = 0.65 * (1 - np.exp(-0.7 * edge_zscore))

        tier_scale = 1.0 if tier == "ATTACK" else 0.6
        final_frac = base_frac * tier_scale * confidence * traj_mult * mm_mult
        shares = max(effective_min_shares, int(size_ceiling * final_frac))

        parts = [f"{tier} tier"]
        if edge_zscore > 0:
            parts.append(f"edge={edge:.2f} ({edge_zscore:.2f} sig)")
        parts.append(f"base={base_frac:.0%}")
        if traj_mult != 1.0:
            parts.append(f"traj={traj_mult:.1f}x [{trajectory}]")
        if mm_mult != 1.0:
            parts.append(f"mm={mm_mult:.1f}x [{mm_confidence}]")
        parts.append(f"-> {final_frac:.0%} of max -> {shares} shares")
        reasoning = " | ".join(parts)

    shares = min(shares, size_ceiling)
    shares = max(shares, effective_min_shares)

    if affordability_limited:
        reasoning += (
            f" Affordability cap: could not fund the 10-share minimum at {trade_price:.2f}, "
            f"so using {shares} shares."
        )

    cost = shares * trade_price
    expected_pnl = shares * edge
    worst_case = shares * cv_rmse

    if capital_ratio < 0.85:
        survival_threshold = 0.10
    elif capital_ratio < 1.0:
        survival_threshold = 0.15
    else:
        survival_threshold = 0.20

    if worst_case > capital * survival_threshold:
        old_shares = shares
        shares = effective_min_shares
        cost = shares * trade_price
        expected_pnl = shares * edge
        worst_case = shares * cv_rmse
        reasoning += (
            f" Survival cap: {old_shares}->{shares} "
            f"(worst case exceeded {survival_threshold:.0%} of capital)"
        )

    running_sortino = None
    if len(pnl_history) >= 2:
        returns = np.array(pnl_history)
        mean_ret = returns.mean()
        downside = returns[returns < 0]
        down_std = np.sqrt(np.mean(downside**2)) if len(downside) > 0 else 1e-6
        running_sortino = round(mean_ret / down_std, 3) if down_std > 0 else None

    return {
        "action": action,
        "shares": shares,
        "trade_price": trade_price,
        "cost": round(cost, 2),
        "edge": round(edge, 2),
        "edge_zscore": round(edge_zscore, 2),
        "confidence": round(confidence, 2),
        "expected_pnl": round(expected_pnl, 2),
        "worst_case": round(worst_case, 2),
        "tier": tier,
        "trajectory": trajectory,
        "mm_confidence": mm_confidence,
        "running_sortino": running_sortino,
        "reasoning": reasoning,
        "affordability_limited": affordability_limited,
    }


def print_decision(stock_num, decision):
    """Pretty-print a trade decision."""
    d = decision
    print(f"\n{'=' * 65}")
    print(f"  STOCK {stock_num} [{d['tier']}] - {d['action']} {d['shares']} shares @ {d['trade_price']:.2f}")
    print(f"{'=' * 65}")
    print(f"  Edge:          {d['edge']:>8.2f}  ({d['edge_zscore']:.2f} sig)")
    print(f"  Confidence:    {d['confidence']:>8.2f}")
    print(f"  Trajectory:    {d['trajectory']:<12}  |  MM confidence: {d['mm_confidence']}")
    print(f"  Cost:          {d['cost']:>10.2f}")
    print(f"  Expected P&L:  {d['expected_pnl']:>+10.2f}")
    print(f"  Worst case:    {d['worst_case']:>10.2f}")
    if d["running_sortino"] is not None:
        print(f"  Sortino (so far): {d['running_sortino']:>6.3f}")
    print(f"  Reasoning:     {d['reasoning']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trade decision helper")
    parser.add_argument("--stock", type=int, help="Stock number (1-9)")
    parser.add_argument("--mm_bid", type=float, help="Market Maker's bid")
    parser.add_argument("--mm_ask", type=float, help="Market Maker's ask")
    parser.add_argument("--capital", type=float, default=100_000, help="Current capital")
    parser.add_argument("--round", type=int, default=1, help="Current round number")
    args = parser.parse_args()

    data_dir = Path(__file__).parent.parent / "hackathon_data"

    if args.stock and args.mm_bid and args.mm_ask:
        stock_number = args.stock
        train = pd.read_csv(data_dir / f"stock_{stock_number}_train.csv")
        test = pd.read_csv(data_dir / f"stock_{stock_number}_test.csv")
        X = train.drop("target", axis=1)
        y = train["target"]

        profile = profile_dataset(train, test)
        result = select_model(profile, X, y, stock_number=stock_number, verbose=False)
        prediction = result["model"].predict(test)[0]

        decision = decide_trade(
            prediction=prediction,
            cv_rmse=result["cv_rmse"],
            mm_bid=args.mm_bid,
            mm_ask=args.mm_ask,
            capital=args.capital,
            noise_ratio=profile.noise_ratio,
            target_mean=profile.target_mean,
            target_std=profile.target_std,
            round_num=args.round,
        )
        print(f"\n  Your prediction: {prediction:.2f}")
        print(f"  MM quotes:       bid={args.mm_bid:.2f}  ask={args.mm_ask:.2f}")
        print_decision(stock_number, decision)
