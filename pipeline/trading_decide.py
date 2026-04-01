"""
Trade Decision Helper for QMML Market Making Hackathon.
Optimised for survival + Sortino ratio + P&L.

Key features:
  - Capital trajectory: more aggressive when up, tighter when down
  - MM spread analysis: wide MM spread = they're uncertain = exploit harder
  - Round awareness: conservative early, aggressive late if ahead
  - Smooth position sizing: continuous function, no abrupt jumps
  - Multi-prize targeting: Sortino-optimal by default, P&L-chasing if ahead

Usage:
    python trade_decide.py                                          # interactive
    python trade_decide.py --stock 3 --mm_bid 240 --mm_ask 260    # single stock
"""

import numpy as np
import argparse


def decide_trade(prediction, cv_rmse, mm_bid, mm_ask, capital,
                 noise_ratio=0.5, target_mean=None, target_std=None,
                 min_shares=10, max_risk_pct=0.15,
                 round_num=1, total_rounds=9, starting_capital=100_000,
                 pnl_history=None):
    """
    Decide whether to buy or sell against the Market Maker.

    Args:
        prediction:       Your model's price estimate
        cv_rmse:          Your model's CV RMSE (uncertainty)
        mm_bid:           Market Maker's bid (you sell to them here)
        mm_ask:           Market Maker's ask (you buy from them here)
        capital:          Your current capital
        noise_ratio:      Model's noise_ratio for this stock
        target_mean:      Training target mean (for mean-reversion fallback)
        target_std:       Training target std (for MM spread analysis)
        min_shares:       Minimum required (rules: 10)
        max_risk_pct:     Max fraction of capital to risk per round
        round_num:        Current round (1-9)
        total_rounds:     Total rounds in competition
        starting_capital: Starting capital for trajectory calc
        pnl_history:      List of P&L from previous rounds (for Sortino tracking)
    """

    if pnl_history is None:
        pnl_history = []

    mm_mid = (mm_bid + mm_ask) / 2
    mm_spread = mm_ask - mm_bid

    # ──────────────────────────────────────────────
    # 0. DETERMINE TIER (mirrors spread.py)
    # ──────────────────────────────────────────────
    if noise_ratio < 0.25:
        tier = "ATTACK"
    elif noise_ratio < 0.5:
        tier = "MODERATE"
    else:
        tier = "SURVIVE"

    # ──────────────────────────────────────────────
    # 1. CAPITAL TRAJECTORY — how are we doing?
    # ──────────────────────────────────────────────
    capital_ratio = capital / starting_capital  # >1 = up, <1 = down
    rounds_left = total_rounds - round_num + 1

    # Trajectory multiplier: scales aggression based on performance
    if capital_ratio > 1.15:
        trajectory = "AHEAD"
        traj_mult = 1.3         # up 15%+ → push harder
    elif capital_ratio > 1.0:
        trajectory = "SLIGHT_UP"
        traj_mult = 1.1         # slightly up → mild boost
    elif capital_ratio > 0.85:
        trajectory = "SLIGHT_DOWN"
        traj_mult = 0.8         # slightly down → pull back
    else:
        trajectory = "DANGER"
        traj_mult = 0.4         # down 15%+ → survival mode
        max_risk_pct = 0.05     # override: hard cap risk at 5%

    # Late-round adjustment: if ahead and few rounds left, lock in gains
    # If behind and few rounds left, take more risk (nothing to lose)
    if rounds_left <= 3:
        if trajectory == "AHEAD":
            traj_mult *= 0.7    # protect gains in final rounds
        elif trajectory == "DANGER":
            traj_mult *= 1.5    # hail mary if we're behind late

    # ──────────────────────────────────────────────
    # 2. MM SPREAD ANALYSIS — what does their spread tell us?
    # ──────────────────────────────────────────────
    # Narrow MM spread = they're confident = riskier to trade against
    # Wide MM spread = they're uncertain = easier to exploit
    mm_confidence = "UNKNOWN"
    mm_mult = 1.0

    if target_std and target_std > 0:
        spread_ratio = mm_spread / target_std  # MM spread as fraction of target volatility
        if spread_ratio < 0.3:
            mm_confidence = "VERY_CONFIDENT"
            mm_mult = 0.5      # they might be right — trade small
        elif spread_ratio < 0.6:
            mm_confidence = "CONFIDENT"
            mm_mult = 0.8      # moderate caution
        elif spread_ratio < 1.0:
            mm_confidence = "UNCERTAIN"
            mm_mult = 1.0      # neutral
        else:
            mm_confidence = "VERY_UNCERTAIN"
            mm_mult = 1.3      # exploit their uncertainty

    # ──────────────────────────────────────────────
    # 3. DIRECTION
    # ──────────────────────────────────────────────
    buy_edge  = prediction - mm_ask    # positive → BUY
    sell_edge = mm_bid - prediction    # positive → SELL

    if buy_edge > 0:
        action = "BUY"
        edge = buy_edge
        trade_price = mm_ask
    elif sell_edge > 0:
        action = "SELL"
        edge = sell_edge
        trade_price = mm_bid
    else:
        # Prediction inside MM spread — no clear model edge
        edge = 0
        if tier == "SURVIVE" and target_mean is not None:
            # Mean-reversion fallback
            if mm_mid > target_mean:
                action = "SELL"
                trade_price = mm_bid
            else:
                action = "BUY"
                trade_price = mm_ask
        else:
            # Trust model direction
            if prediction >= mm_mid:
                action = "BUY"
                trade_price = mm_ask
            else:
                action = "SELL"
                trade_price = mm_bid

    # ──────────────────────────────────────────────
    # 4. POSITION SIZING — smooth, tier-dependent
    # ──────────────────────────────────────────────
    edge_zscore = edge / cv_rmse if cv_rmse > 0 else 0
    confidence = max(0.05, 1.0 - noise_ratio)
    max_risk = capital * max_risk_pct
    max_shares_by_capital = int(max_risk / trade_price) if trade_price > 0 else min_shares

    if tier == "SURVIVE":
        # Always minimum — backtest confirmed negative P&L on these
        shares = min_shares
        reasoning = f"SURVIVE (noise={noise_ratio:.2f}) — minimum 10 shares."
        if edge == 0 and target_mean is not None:
            direction = "above" if action == "SELL" else "below"
            reasoning += f" Mean-reversion: MM mid {direction} target mean."

    else:
        # ATTACK or MODERATE — smooth sizing function
        # Base fraction: sigmoid-like curve on edge_zscore
        # 0σ → 0%, 0.5σ → ~10%, 1σ → ~25%, 2σ → ~50%, 3σ+ → ~65%
        if edge_zscore <= 0:
            base_frac = 0.0
        else:
            base_frac = 0.65 * (1 - np.exp(-0.7 * edge_zscore))

        # Scale by tier
        if tier == "ATTACK":
            tier_scale = 1.0
        else:  # MODERATE
            tier_scale = 0.6

        # Apply all multipliers
        final_frac = base_frac * tier_scale * confidence * traj_mult * mm_mult

        # Convert to shares
        shares = max(min_shares, int(max_shares_by_capital * final_frac))

        # Build reasoning
        parts = [f"{tier} tier"]
        if edge_zscore > 0:
            parts.append(f"edge={edge:.2f} ({edge_zscore:.2f}σ)")
        parts.append(f"base={base_frac:.0%}")
        if traj_mult != 1.0:
            parts.append(f"traj={traj_mult:.1f}x [{trajectory}]")
        if mm_mult != 1.0:
            parts.append(f"mm={mm_mult:.1f}x [{mm_confidence}]")
        parts.append(f"→ {final_frac:.0%} of max → {shares} shares")
        reasoning = " | ".join(parts)

    # Clamp
    shares = min(shares, max_shares_by_capital)
    shares = max(shares, min_shares)

    # P&L estimates
    cost = shares * trade_price
    expected_pnl = shares * edge
    worst_case = shares * cv_rmse

    # ──────────────────────────────────────────────
    # 5. SURVIVAL CHECK — never risk elimination
    # ──────────────────────────────────────────────
    # Adaptive threshold: tighter when capital is low
    if capital_ratio < 0.85:
        survival_threshold = 0.10  # only risk 10% when in danger
    elif capital_ratio < 1.0:
        survival_threshold = 0.15
    else:
        survival_threshold = 0.20

    if worst_case > capital * survival_threshold:
        old_shares = shares
        shares = min_shares
        cost = shares * trade_price
        expected_pnl = shares * edge
        worst_case = shares * cv_rmse
        reasoning += f" ⚠️ Survival cap: {old_shares}→{shares} (worst case exceeded {survival_threshold:.0%} of capital)"

    # ──────────────────────────────────────────────
    # 6. SORTINO TRACKING — running estimate
    # ──────────────────────────────────────────────
    running_sortino = None
    if len(pnl_history) >= 2:
        returns = np.array(pnl_history)
        mean_ret = returns.mean()
        downside = returns[returns < 0]
        down_std = np.sqrt(np.mean(downside**2)) if len(downside) > 0 else 1e-6
        running_sortino = round(mean_ret / down_std, 3) if down_std > 0 else None

    return {
        "action":           action,
        "shares":           shares,
        "trade_price":      trade_price,
        "cost":             round(cost, 2),
        "edge":             round(edge, 2),
        "edge_zscore":      round(edge_zscore, 2),
        "confidence":       round(confidence, 2),
        "expected_pnl":     round(expected_pnl, 2),
        "worst_case":       round(worst_case, 2),
        "tier":             tier,
        "trajectory":       trajectory,
        "mm_confidence":    mm_confidence,
        "running_sortino":  running_sortino,
        "reasoning":        reasoning,
    }


def print_decision(stock_num, decision):
    """Pretty-print a trade decision."""
    d = decision
    print(f"\n{'='*65}")
    print(f"  STOCK {stock_num} [{d['tier']}] — {d['action']} {d['shares']} shares @ {d['trade_price']:.2f}")
    print(f"{'='*65}")
    print(f"  Edge:          {d['edge']:>8.2f}  ({d['edge_zscore']:.2f}σ)")
    print(f"  Confidence:    {d['confidence']:>8.2f}")
    print(f"  Trajectory:    {d['trajectory']:<12}  |  MM confidence: {d['mm_confidence']}")
    print(f"  Cost:          {d['cost']:>10.2f}")
    print(f"  Expected P&L:  {d['expected_pnl']:>+10.2f}")
    print(f"  Worst case:    {d['worst_case']:>10.2f}")
    if d['running_sortino'] is not None:
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

    from pathlib import Path
    import sys
    import pandas as pd
    sys.path.append(str(Path(__file__).parent))
    from data_check import profile_dataset
    from model_select import select_model

    DATA_DIR = Path(__file__).parent.parent / "hackathon_data"

    if args.stock and args.mm_bid and args.mm_ask:
        # Single stock mode
        i = args.stock
        train = pd.read_csv(DATA_DIR / f"stock_{i}_train.csv")
        test  = pd.read_csv(DATA_DIR / f"stock_{i}_test.csv")
        X = train.drop("target", axis=1)
        y = train["target"]

        profile = profile_dataset(train, test)
        result  = select_model(profile, X, y, stock_number=i, verbose=False)
        pred    = result["model"].predict(test)[0]

        decision = decide_trade(
            prediction=pred,
            cv_rmse=result["cv_rmse"],
            mm_bid=args.mm_bid,
            mm_ask=args.mm_ask,
            capital=args.capital,
            noise_ratio=profile.noise_ratio,
            target_mean=profile.target_mean,
            target_std=profile.target_std,
            round_num=args.round,
        )
        print(f"\n  Your prediction: {pred:.2f}")
        print(f"  MM quotes:       bid={args.mm_bid:.2f}  ask={args.mm_ask:.2f}")
        print_decision(i, decision)
    else:
        # Interactive mode — pre-compute all predictions first
        starting_capital = 100_000
        capital = starting_capital
        pnl_history = []

        print("\n  QMML Trade Decision Helper (Optimised)")
        print("  Pre-computing all models... (this takes a few minutes)\n")

        # --- PHASE 1: Pre-compute all models ---
        stock_cache = {}
        for i in range(1, 10):
            print(f"  Training stock {i}...", end=" ", flush=True)
            train = pd.read_csv(DATA_DIR / f"stock_{i}_train.csv")
            test  = pd.read_csv(DATA_DIR / f"stock_{i}_test.csv")
            X = train.drop("target", axis=1)
            y = train["target"]

            profile = profile_dataset(train, test)
            result  = select_model(profile, X, y, stock_number=i, verbose=False)
            pred    = result["model"].predict(test)[0]

            stock_cache[i] = {
                "pred": pred,
                "cv_rmse": result["cv_rmse"],
                "noise_ratio": profile.noise_ratio,
                "target_mean": profile.target_mean,
                "target_std": profile.target_std,
                "model_name": result["name"],
            }
            tier = "ATTACK" if profile.noise_ratio < 0.25 else ("MODERATE" if profile.noise_ratio < 0.5 else "SURVIVE")
            print(f"done  [{tier}] pred={pred:.2f}")

        print(f"\n  All models ready. Starting capital: {capital:.2f}")
        print(f"{'='*65}")

        # --- PHASE 2: Live decisions ---
        for i in range(1, 10):
            c = stock_cache[i]
            tier = "ATTACK" if c["noise_ratio"] < 0.25 else ("MODERATE" if c["noise_ratio"] < 0.5 else "SURVIVE")
            cap_pct = (capital / starting_capital - 1) * 100

            print(f"\n  --- Round {i}/9 [{tier}] | pred: {c['pred']:.2f} | model: {c['model_name']} | capital: {capital:.2f} ({cap_pct:+.1f}%) ---")
            raw = input(f"  MM bid,ask (e.g. '240,260' or 'skip'): ").strip()

            if raw.lower() == "skip":
                print("  Skipped.")
                pnl_history.append(0.0)
                continue

            try:
                mm_bid, mm_ask = [float(x.strip()) for x in raw.split(",")]
            except ValueError:
                print("  Invalid input, skipping.")
                pnl_history.append(0.0)
                continue

            decision = decide_trade(
                prediction=c["pred"],
                cv_rmse=c["cv_rmse"],
                mm_bid=mm_bid,
                mm_ask=mm_ask,
                capital=capital,
                noise_ratio=c["noise_ratio"],
                target_mean=c["target_mean"],
                target_std=c["target_std"],
                round_num=i,
                total_rounds=9,
                starting_capital=starting_capital,
                pnl_history=pnl_history,
            )
            print_decision(i, decision)

            print(f"\n  Current capital: {capital:.2f}")
            actual = input(f"  True price (or 'skip'): ").strip()
            if actual.lower() != "skip":
                try:
                    true_price = float(actual)
                    if decision["action"] == "BUY":
                        pnl = decision["shares"] * (true_price - decision["trade_price"])
                    else:
                        pnl = decision["shares"] * (decision["trade_price"] - true_price)
                    capital += pnl
                    pnl_history.append(pnl)
                    print(f"  P&L this round: {pnl:+.2f}")
                    print(f"  New capital:    {capital:.2f}")

                    # Running Sortino
                    if len(pnl_history) >= 2:
                        returns = np.array(pnl_history)
                        down = returns[returns < 0]
                        down_std = np.sqrt(np.mean(down**2)) if len(down) > 0 else 0
                        sortino = returns.mean() / down_std if down_std > 0 else float('inf')
                        print(f"  Running Sortino: {sortino:.3f}")
                except ValueError:
                    pnl_history.append(0.0)
            else:
                pnl_history.append(0.0)

        # Final summary
        print(f"\n{'='*65}")
        print(f"  COMPETITION SUMMARY")
        print(f"{'='*65}")
        print(f"  Starting capital: {starting_capital:.2f}")
        print(f"  Final capital:    {capital:.2f}")
        print(f"  Total P&L:        {capital - starting_capital:+.2f}")
        print(f"  Return:           {(capital/starting_capital - 1)*100:+.1f}%")
        if len([p for p in pnl_history if p != 0]) >= 2:
            returns = np.array(pnl_history)
            down = returns[returns < 0]
            down_std = np.sqrt(np.mean(down**2)) if len(down) > 0 else 0
            sortino = returns.mean() / down_std if down_std > 0 else float('inf')
            print(f"  Final Sortino:    {sortino:.3f}")
        print(f"  Rounds played:    {len([p for p in pnl_history if p != 0])}/9")
