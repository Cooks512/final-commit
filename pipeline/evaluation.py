"""
Evaluation / Backtest for QMML Market Making Pipeline.

Since test CSVs have no target values, we simulate the competition
using leave-one-out (or leave-k-out) on training data.

For each stock:
  - Hold out one row as the "test" point (we know its true price).
  - Train on the remaining rows.
  - Generate bid/ask quotes via the full pipeline.
  - Check: did the true price land inside the spread?
  - Simulate P&L under both MM and trader roles.

Metrics:
  - Coverage: % of rounds where true price is inside [bid, ask]
  - Mean spread width
  - Simulated MM P&L (worst case: every team trades against you)
  - Simulated Trader P&L (you trade against a hypothetical MM)
  - Sortino ratio estimate
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).parent))
from data_check import profile_dataset
from model_select import select_model
from spread import compute_spread


def backtest_stock(train_df, n_trials=None, seed=42, verbose=True):
    """
    Run leave-one-out backtest on a single stock's training data.
    
    Args:
        train_df:  Full training DataFrame (with 'target' column)
        n_trials:  Number of holdout trials (None = all rows, capped at 200)
        seed:      Random seed for reproducibility
    
    Returns:
        dict of evaluation metrics
    """
    rng = np.random.RandomState(seed)
    n = len(train_df)
    
    if n_trials is None:
        n_trials = min(n, 200)  # cap for speed
    
    n_trials = min(n_trials, n)  # can't sample more than we have
    
    indices = rng.choice(n, size=n_trials, replace=False)
    
    results = []
    
    for trial_idx, holdout_idx in enumerate(indices):
        # Split: holdout one row as "test", rest as "train"
        test_row = train_df.iloc[[holdout_idx]].copy()
        train_subset = train_df.drop(index=train_df.index[holdout_idx]).reset_index(drop=True)
        
        true_price = float(test_row["target"].iloc[0])
        test_features = test_row.drop(columns=["target"])
        
        # Run pipeline
        try:
            profile = profile_dataset(train_subset, test_features)
            X_train = train_subset.drop(columns=["target"])
            y_train = train_subset["target"]
            
            result = select_model(profile, X_train, y_train, verbose=False)
            
            pred = float(result["model"].predict(test_features)[0])
            
            # Overfit ratio
            train_pred = result["model"].predict(X_train)
            train_rmse = float(np.sqrt(np.mean((train_pred - y_train.values)**2)))
            overfit_ratio = train_rmse / result["cv_rmse"] if result["cv_rmse"] > 0 else 1.0
            
            quotes = compute_spread(
                pred, result["cv_rmse"], profile,
                overfit_ratio=overfit_ratio,
            )
            
            bid = quotes["bid"]
            ask = quotes["ask"]
            spread = quotes["spread"]
            role = quotes["role"]
            
            # Evaluation
            inside = bid <= true_price <= ask
            pred_error = abs(pred - true_price)
            
            # --- Simulated P&L ---
            
            # As MM: assume 5 teams each trade 10 shares against you.
            # They trade optimally: buy at your ask if true > ask,
            # sell at your bid if true < bid.
            n_opponents = 5
            shares_each = 10
            mm_pnl = 0.0
            if true_price > ask:
                # Opponents buy at your ask (you sell at ask, true > ask → you lose)
                mm_pnl = n_opponents * shares_each * (ask - true_price)  # negative
            elif true_price < bid:
                # Opponents sell at your bid (you buy at bid, true < bid → you lose)
                mm_pnl = n_opponents * shares_each * (true_price - bid)  # negative
            else:
                # True price inside spread — some opponents guess wrong
                # Approximate: half buy (lose), half sell (lose) → you profit from spread
                mm_pnl = n_opponents * shares_each * (spread / 4)  # rough estimate
            
            # As Trader: you trade 10 shares against a hypothetical MM
            # with a moderately tight spread centered on the mean.
            hyp_mm_mid = float(y_train.mean())
            hyp_mm_half = float(y_train.std() * 0.3)  # hypothetical tight MM
            hyp_mm_bid = hyp_mm_mid - hyp_mm_half
            hyp_mm_ask = hyp_mm_mid + hyp_mm_half
            
            trader_shares = 10
            if pred > hyp_mm_ask:
                # We buy at their ask
                trader_pnl = trader_shares * (true_price - hyp_mm_ask)
            elif pred < hyp_mm_bid:
                # We sell at their bid
                trader_pnl = trader_shares * (hyp_mm_bid - true_price)
            else:
                # Our pred is inside their spread — minimal trade
                if pred >= hyp_mm_mid:
                    trader_pnl = trader_shares * (true_price - hyp_mm_ask)
                else:
                    trader_pnl = trader_shares * (hyp_mm_bid - true_price)
            
            results.append({
                "trial":        trial_idx,
                "true_price":   true_price,
                "prediction":   pred,
                "pred_error":   pred_error,
                "bid":          bid,
                "ask":          ask,
                "spread":       spread,
                "inside":       inside,
                "role":         role,
                "mm_pnl":       mm_pnl,
                "trader_pnl":   trader_pnl,
                "noise_ratio":  profile.noise_ratio,
                "cv_rmse":      result["cv_rmse"],
                "model":        result["name"],
            })
        except Exception as e:
            if verbose:
                print(f"    Trial {trial_idx} failed: {e}")
            continue
    
    if not results:
        return None
    
    df = pd.DataFrame(results)
    
    # --- Aggregate metrics ---
    coverage = df["inside"].mean() * 100
    mean_spread = df["spread"].mean()
    median_spread = df["spread"].median()
    mean_pred_error = df["pred_error"].mean()
    
    # MM metrics
    mm_total_pnl = df["mm_pnl"].sum()
    mm_mean_pnl = df["mm_pnl"].mean()
    mm_win_rate = (df["mm_pnl"] > 0).mean() * 100
    
    # Trader metrics
    trader_total_pnl = df["trader_pnl"].sum()
    trader_mean_pnl = df["trader_pnl"].mean()
    trader_win_rate = (df["trader_pnl"] > 0).mean() * 100
    
    # Sortino ratio (on trader P&L per round)
    returns = df["trader_pnl"].values
    mean_return = returns.mean()
    downside = returns[returns < 0]
    downside_std = np.sqrt(np.mean(downside**2)) if len(downside) > 0 else 1e-6
    sortino = mean_return / downside_std if downside_std > 0 else 0.0
    
    # Attack vs Moderate vs Survive breakdown
    attack_df   = df[df["role"] == "ATTACK"]
    moderate_df = df[df["role"] == "MODERATE"]
    survive_df  = df[df["role"] == "SURVIVE"]
    
    metrics = {
        "n_trials":         len(df),
        "coverage_pct":     round(coverage, 1),
        "mean_spread":      round(mean_spread, 2),
        "median_spread":    round(median_spread, 2),
        "mean_pred_error":  round(mean_pred_error, 2),
        "mm_total_pnl":     round(mm_total_pnl, 2),
        "mm_mean_pnl":      round(mm_mean_pnl, 2),
        "mm_win_rate":      round(mm_win_rate, 1),
        "trader_total_pnl": round(trader_total_pnl, 2),
        "trader_mean_pnl":  round(trader_mean_pnl, 2),
        "trader_win_rate":  round(trader_win_rate, 1),
        "sortino":          round(sortino, 3),
        "attack_trials":    len(attack_df),
        "moderate_trials":  len(moderate_df),
        "survive_trials":   len(survive_df),
        "attack_coverage":  round(attack_df["inside"].mean() * 100, 1) if len(attack_df) > 0 else None,
        "moderate_coverage": round(moderate_df["inside"].mean() * 100, 1) if len(moderate_df) > 0 else None,
        "survive_coverage": round(survive_df["inside"].mean() * 100, 1) if len(survive_df) > 0 else None,
    }
    
    return metrics


def run_full_backtest(data_dir, n_trials_per_stock=100, verbose=True):
    """Run backtest across all 9 stocks."""
    
    data_dir = Path(data_dir)
    all_metrics = {}
    
    if verbose:
        print("\n" + "=" * 90)
        print("  PIPELINE BACKTEST — Simulating competition rounds on training data")
        print("=" * 90)
    
    for i in range(1, 10):
        train_path = data_dir / f"stock_{i}_train.csv"
        if not train_path.exists():
            if verbose:
                print(f"\n  Stock {i}: train file not found, skipping.")
            continue
        
        train = pd.read_csv(train_path)
        
        if verbose:
            print(f"\n  Stock {i} ({len(train)} rows, {n_trials_per_stock} trials)...", end=" ", flush=True)
        
        metrics = backtest_stock(train, n_trials=n_trials_per_stock, verbose=False)
        
        if metrics is None:
            if verbose:
                print("FAILED")
            continue
        
        all_metrics[i] = metrics
        
        if verbose:
            m = metrics
            print(f"done")
            print(f"    Coverage:    {m['coverage_pct']:>5.1f}%   |  Mean spread: {m['mean_spread']:>7.2f}  |  Mean error: {m['mean_pred_error']:>7.2f}")
            print(f"    MM P&L:      {m['mm_mean_pnl']:>+8.2f}/round  (win rate: {m['mm_win_rate']:>5.1f}%)")
            print(f"    Trader P&L:  {m['trader_mean_pnl']:>+8.2f}/round  (win rate: {m['trader_win_rate']:>5.1f}%)")
            print(f"    Sortino:     {m['sortino']:>7.3f}   |  Role split: {m['attack_trials']} ATTACK / {m['moderate_trials']} MODERATE / {m['survive_trials']} SURVIVE")
            atk = f"{m['attack_coverage']:.1f}%" if m['attack_coverage'] is not None else "N/A"
            mod = f"{m['moderate_coverage']:.1f}%" if m['moderate_coverage'] is not None else "N/A"
            srv = f"{m['survive_coverage']:.1f}%" if m['survive_coverage'] is not None else "N/A"
            print(f"    ATTACK: {atk}  |  MODERATE: {mod}  |  SURVIVE: {srv}")
    
    # --- Summary ---
    if verbose and all_metrics:
        print(f"\n{'='*90}")
        print(f"  SUMMARY ACROSS ALL STOCKS")
        print(f"{'='*90}")
        
        all_coverage = [m["coverage_pct"] for m in all_metrics.values()]
        all_trader_pnl = [m["trader_mean_pnl"] for m in all_metrics.values()]
        all_sortino = [m["sortino"] for m in all_metrics.values()]
        all_mm_pnl = [m["mm_mean_pnl"] for m in all_metrics.values()]
        
        print(f"\n  {'Stock':<8} {'Coverage':>10} {'Spread':>10} {'Error':>10} {'MM P&L':>10} {'Trader P&L':>12} {'Sortino':>10} {'Role':>10}")
        print(f"  {'-'*82}")
        for i, m in sorted(all_metrics.items()):
            counts = {"ATTACK": m["attack_trials"], "MODERATE": m["moderate_trials"], "SURVIVE": m["survive_trials"]}
            role = max(counts, key=counts.get)
            print(f"  {i:<8} {m['coverage_pct']:>9.1f}% {m['mean_spread']:>10.2f} {m['mean_pred_error']:>10.2f} "
                  f"{m['mm_mean_pnl']:>+10.2f} {m['trader_mean_pnl']:>+12.2f} {m['sortino']:>10.3f} {role:>10}")
        
        print(f"\n  Avg coverage:    {np.mean(all_coverage):.1f}%")
        print(f"  Avg trader P&L:  {np.mean(all_trader_pnl):+.2f} per round")
        print(f"  Avg Sortino:     {np.mean(all_sortino):.3f}")
        print(f"  Avg MM P&L:      {np.mean(all_mm_pnl):+.2f} per round")
        
        # Highlight best and worst
        best_stock = max(all_metrics.items(), key=lambda kv: kv[1]["trader_mean_pnl"])
        worst_stock = min(all_metrics.items(), key=lambda kv: kv[1]["trader_mean_pnl"])
        print(f"\n  Best trader stock:  {best_stock[0]} ({best_stock[1]['trader_mean_pnl']:+.2f}/round)")
        print(f"  Worst trader stock: {worst_stock[0]} ({worst_stock[1]['trader_mean_pnl']:+.2f}/round)")
    
    return all_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Backtest the pipeline on training data")
    parser.add_argument("--trials", type=int, default=100, help="Holdout trials per stock (default 100)")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to hackathon_data folder")
    args = parser.parse_args()
    
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = Path(__file__).parent.parent / "hackathon_data"
    
    run_full_backtest(data_dir, n_trials_per_stock=args.trials)