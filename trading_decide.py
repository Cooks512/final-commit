"""
Trade Decision Helper for QMML Market Making Hackathon.

Usage during live rounds:
    python trade_decide.py --stock 3 --mm_bid 240 --mm_ask 260 --capital 95000

Or interactive mode:
    python trade_decide.py

Logic:
  - If your prediction > MM's ask → BUY (you think it's worth more than they're selling for)
  - If your prediction < MM's bid → SELL (you think it's worth less than they're buying for)
  - If prediction is between bid and ask → MM's spread covers your prediction, MINIMUM trade only
  - Share sizing based on edge magnitude and confidence
"""

import numpy as np
import argparse


def decide_trade(prediction, cv_rmse, mm_bid, mm_ask, capital,
                 noise_ratio=0.5, min_shares=10, max_risk_pct=0.15):
    """
    Decide whether to buy or sell against the Market Maker.
    
    Args:
        prediction:    Your model's price estimate
        cv_rmse:       Your model's CV RMSE (uncertainty)
        mm_bid:        Market Maker's bid (price they buy at → you sell at)
        mm_ask:        Market Maker's ask (price they sell at → you buy at)
        capital:       Your current capital
        noise_ratio:   Your model's noise_ratio for this stock
        min_shares:    Minimum shares required (rules say 10)
        max_risk_pct:  Max fraction of capital to risk per round
    
    Returns:
        dict with action, shares, reasoning
    """

    mm_mid = (mm_bid + mm_ask) / 2
    mm_spread = mm_ask - mm_bid

    # ──────────────────────────────────────────────
    # 1. DIRECTION: buy, sell, or neutral?
    # ──────────────────────────────────────────────
    # Edge = how far our prediction is from the MM's price
    buy_edge  = prediction - mm_ask    # positive → we think true price > ask → BUY
    sell_edge = mm_bid - prediction    # positive → we think true price < bid → SELL

    if buy_edge > 0:
        action = "BUY"
        edge = buy_edge
        trade_price = mm_ask
    elif sell_edge > 0:
        action = "SELL"
        edge = sell_edge
        trade_price = mm_bid
    else:
        # Our prediction is inside their spread — no clear edge
        # Still must trade minimum, so pick the side with more room
        if prediction >= mm_mid:
            action = "BUY"
            edge = 0
            trade_price = mm_ask
        else:
            action = "SELL"
            edge = 0
            trade_price = mm_bid

    # ──────────────────────────────────────────────
    # 2. CONFIDENCE: how sure are we about the edge?
    # ──────────────────────────────────────────────
    # Edge in units of RMSE — how many "standard errors" of edge we have
    edge_zscore = edge / cv_rmse if cv_rmse > 0 else 0

    # Confidence discount based on noise ratio
    # noise_ratio near 0 → full confidence; near 1 → minimal confidence
    confidence = max(0.05, 1.0 - noise_ratio)

    # ──────────────────────────────────────────────
    # 3. POSITION SIZING
    # ──────────────────────────────────────────────
    max_risk = capital * max_risk_pct
    max_shares_by_capital = int(max_risk / trade_price) if trade_price > 0 else min_shares

    if edge_zscore <= 0:
        # No edge — trade minimum only
        shares = min_shares
        reasoning = "No clear edge — trading minimum required shares."
    elif edge_zscore < 0.5:
        # Small edge — slightly above minimum
        shares = int(min_shares * 1.5)
        reasoning = f"Small edge ({edge:.2f}, {edge_zscore:.2f}σ) — slight overweight."
    elif edge_zscore < 1.0:
        # Decent edge — moderate size
        frac = 0.3 * confidence
        shares = max(min_shares, int(max_shares_by_capital * frac))
        reasoning = f"Moderate edge ({edge:.2f}, {edge_zscore:.2f}σ) — sizing at {frac:.0%} of max."
    elif edge_zscore < 2.0:
        # Strong edge — larger size
        frac = 0.5 * confidence
        shares = max(min_shares, int(max_shares_by_capital * frac))
        reasoning = f"Strong edge ({edge:.2f}, {edge_zscore:.2f}σ) — sizing at {frac:.0%} of max."
    else:
        # Very strong edge — go big (but capped)
        frac = 0.7 * confidence
        shares = max(min_shares, int(max_shares_by_capital * frac))
        reasoning = f"Very strong edge ({edge:.2f}, {edge_zscore:.2f}σ) — sizing at {frac:.0%} of max."

    # Hard cap: never risk more than max_risk_pct of capital
    shares = min(shares, max_shares_by_capital)
    shares = max(shares, min_shares)

    # Cost and potential P&L
    cost = shares * trade_price
    expected_pnl = shares * edge  # if our prediction is exactly right
    worst_case = shares * cv_rmse  # rough worst-case loss

    # ──────────────────────────────────────────────
    # 4. SURVIVAL CHECK
    # ──────────────────────────────────────────────
    # If this trade could wipe us out, scale back to minimum
    if worst_case > capital * 0.25:
        shares = min_shares
        cost = shares * trade_price
        expected_pnl = shares * edge
        worst_case = shares * cv_rmse
        reasoning += " ⚠️ Scaled to minimum — worst case exceeds 25% of capital."

    return {
        "action":        action,
        "shares":        shares,
        "trade_price":   trade_price,
        "cost":          round(cost, 2),
        "edge":          round(edge, 2),
        "edge_zscore":   round(edge_zscore, 2),
        "confidence":    round(confidence, 2),
        "expected_pnl":  round(expected_pnl, 2),
        "worst_case":    round(worst_case, 2),
        "reasoning":     reasoning,
    }


def print_decision(stock_num, decision):
    """Pretty-print a trade decision."""
    d = decision
    print(f"\n{'='*55}")
    print(f"  STOCK {stock_num} — {d['action']} {d['shares']} shares @ {d['trade_price']:.2f}")
    print(f"{'='*55}")
    print(f"  Edge:          {d['edge']:>8.2f}  ({d['edge_zscore']:.2f}σ)")
    print(f"  Confidence:    {d['confidence']:>8.2f}")
    print(f"  Cost:          {d['cost']:>10.2f}")
    print(f"  Expected P&L:  {d['expected_pnl']:>+10.2f}")
    print(f"  Worst case:    {d['worst_case']:>10.2f}")
    print(f"  Reasoning:     {d['reasoning']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trade decision helper")
    parser.add_argument("--stock", type=int, help="Stock number (1-9)")
    parser.add_argument("--mm_bid", type=float, help="Market Maker's bid")
    parser.add_argument("--mm_ask", type=float, help="Market Maker's ask")
    parser.add_argument("--capital", type=float, default=100_000, help="Current capital")
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
        stocks = [args.stock]
    else:
        # Interactive mode — prompt for each
        stocks = None

    if stocks:
        for i in stocks:
            train = pd.read_csv(DATA_DIR / f"stock_{i}_train.csv")
            test  = pd.read_csv(DATA_DIR / f"stock_{i}_test.csv")
            X = train.drop("target", axis=1)
            y = train["target"]

            profile = profile_dataset(train, test)
            result  = select_model(profile, X, y, verbose=False)
            pred    = result["model"].predict(test)[0]

            decision = decide_trade(
                prediction=pred,
                cv_rmse=result["cv_rmse"],
                mm_bid=args.mm_bid,
                mm_ask=args.mm_ask,
                capital=args.capital,
                noise_ratio=profile.noise_ratio,
            )
            print(f"\n  Your prediction: {pred:.2f}")
            print(f"  MM quotes:       bid={args.mm_bid:.2f}  ask={args.mm_ask:.2f}")
            print_decision(i, decision)
    else:
        # Interactive: run all stocks, ask for MM quotes per round
        capital = 100_000
        print("\n  QMML Trade Decision Helper")
        print("  Enter MM bid/ask for each round (or 'skip' to skip)\n")

        for i in range(1, 10):
            train = pd.read_csv(DATA_DIR / f"stock_{i}_train.csv")
            test  = pd.read_csv(DATA_DIR / f"stock_{i}_test.csv")
            X = train.drop("target", axis=1)
            y = train["target"]

            profile = profile_dataset(train, test)
            result  = select_model(profile, X, y, verbose=False)
            pred    = result["model"].predict(test)[0]

            print(f"\n  --- Round {i} (your pred: {pred:.2f}, noise: {profile.noise_ratio:.2f}) ---")
            raw = input(f"  MM bid,ask for stock {i} (e.g. '240,260' or 'skip'): ").strip()

            if raw.lower() == "skip":
                print("  Skipped.")
                continue

            try:
                mm_bid, mm_ask = [float(x.strip()) for x in raw.split(",")]
            except ValueError:
                print("  Invalid input, skipping.")
                continue

            decision = decide_trade(
                prediction=pred,
                cv_rmse=result["cv_rmse"],
                mm_bid=mm_bid,
                mm_ask=mm_ask,
                capital=capital,
                noise_ratio=profile.noise_ratio,
            )
            print_decision(i, decision)

            # Update capital estimate (rough)
            print(f"\n  Current capital: {capital:.2f}")
            actual = input(f"  Enter true price when revealed (or 'skip'): ").strip()
            if actual.lower() != "skip":
                try:
                    true_price = float(actual)
                    if decision["action"] == "BUY":
                        pnl = decision["shares"] * (true_price - decision["trade_price"])
                    else:
                        pnl = decision["shares"] * (decision["trade_price"] - true_price)
                    capital += pnl
                    print(f"  P&L this round: {pnl:+.2f}")
                    print(f"  New capital:    {capital:.2f}")
                except ValueError:
                    pass

        print(f"\n  Final capital: {capital:.2f}")