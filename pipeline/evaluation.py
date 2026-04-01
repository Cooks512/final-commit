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

import pickle
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression

sys.path.append(str(Path(__file__).parent))
from data_check import profile_dataset
from model_select import select_model
from spread import compute_spread


DEFAULT_DATA_DIR = Path(__file__).parent.parent / "hackathon_data"
DEFAULT_SAVED_MODELS_DIR = Path(__file__).parent.parent / "saved_models"

#  Resolve the data directory, allowing for an optional override via argument
def _resolve_data_dir(data_dir=None):
    return Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR

# Picks which rows to test cases during evaluation.
def _sample_holdout_indices(n_rows, n_trials=None, seed=42):
    rng = np.random.RandomState(seed)
    if n_trials is None:
        n_trials = min(n_rows, 200)
    n_trials = min(n_trials, n_rows)
    return rng.choice(n_rows, size=n_trials, replace=False)

# Build a simple linear regression baseline model for comparison in the backtest.
def _build_linear_baseline():
    return LinearRegression()

# Summarize prediction errors into key metrics.
def _summarize_prediction_errors(errors):
    error_values = np.asarray(errors, dtype=float)
    abs_errors = np.abs(error_values)
    return {
        "n_trials": int(len(error_values)),
        "rmse": float(np.sqrt(np.mean(error_values ** 2))),
        "mae": float(np.mean(abs_errors)),
        "median_abs_error": float(np.median(abs_errors)),
        "mean_error": float(np.mean(error_values)),
    }

# Package the model summary info (excluding the model object itself) for saving.
def _model_package(result):
    return {key: value for key, value in result.items() if key != "model"}

# Save the trained selector model and metadata to disk for later reuse.
def save_selector_model_artifact(
    stock_number,
    train_df,
    test_df,
    selector_result,
    save_dir=None,
):
    """
    Persist the fully trained selector model and the metadata needed to reuse it.
    """
    save_dir = Path(save_dir) if save_dir is not None else DEFAULT_SAVED_MODELS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = train_df.drop(columns=["target"]).columns.tolist()
    test_predictions = selector_result["model"].predict(test_df)
    artifact = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_version": 1,
        "stock_number": int(stock_number),
        "feature_columns": feature_columns,
        "target_column": "target",
        "model": selector_result["model"],
        "model_summary": _model_package(selector_result),
        "train_rows": int(len(train_df)),
        "train_columns": train_df.columns.tolist(),
        "test_rows": int(len(test_df)),
        "test_columns": test_df.columns.tolist(),
        "test_prediction": test_predictions.tolist(),
    }

    artifact_path = save_dir / f"stock_{int(stock_number)}_selector.pkl"
    with open(artifact_path, "wb") as handle:
        pickle.dump(artifact, handle)
    return artifact_path

# Load a saved selector model artifact from disk.
def load_selector_model_artifact(model_path):
    with open(model_path, "rb") as handle:
        return pickle.load(handle)

# Main backtest function for a single stock, using leave-one-out on training data.
def backtest_stock(train_df, stock_number=None, n_trials=None, seed=42, verbose=True):
    """
    Run leave-one-out backtest on a single stock's training data.
    
    Args:
        train_df:  Full training DataFrame (with 'target' column)
        n_trials:  Number of holdout trials (None = all rows, capped at 200)
        seed:      Random seed for reproducibility
    
    Returns:
        dict of evaluation metrics
    """
    n = len(train_df)
    indices = _sample_holdout_indices(n, n_trials=n_trials, seed=seed)
    
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
            
            result = select_model(
                profile,
                X_train,
                y_train,
                stock_number=stock_number,
                verbose=False,
            )
            
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

# Main function to test the selector against a linear baseline and save the model if it wins.
def evaluate_stock_selector_vs_linear(
    stock_number,
    data_dir=None,
    n_trials=100,
    seed=42,
    save_dir=None,
    verbose=True,
):
    """
    Compare the selector against a plain LinearRegression baseline on one stock.

    The comparison uses the same holdout rows for both models. If the selector
    achieves lower holdout RMSE than the baseline, it is then trained on the
    full stock train set and saved to disk together with the metadata needed
    to reload and rerun it.
    """

    # Validate stock number
    stock_number = int(stock_number)
    if stock_number < 1 or stock_number > 9:
        raise ValueError("stock_number must be between 1 and 9")

    # Load data and sample holdout indices
    data_dir = _resolve_data_dir(data_dir)
    train_path = data_dir / f"stock_{stock_number}_train.csv"
    test_path = data_dir / f"stock_{stock_number}_test.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Sample holdout indices
    indices = _sample_holdout_indices(len(train_df), n_trials=n_trials, seed=seed)

    if verbose:
        print("\n" + "=" * 90)
        print(f"  STOCK {stock_number} SELECTOR VS BASIC LINEAR REGRESSION")
        print("=" * 90)
        print(f"  Holdout trials: {len(indices)}")

    # Run trials comparing selector to linear baseline on the same holdout rows
    trial_results = []
    failed_trials = []
    for trial_idx, holdout_idx in enumerate(indices):
        holdout_row = train_df.iloc[[holdout_idx]].copy()
        train_subset = train_df.drop(index=train_df.index[holdout_idx]).reset_index(drop=True)

        true_price = float(holdout_row["target"].iloc[0])
        test_features = holdout_row.drop(columns=["target"])
        X_train = train_subset.drop(columns=["target"])
        y_train = train_subset["target"]

        try:
            selector_profile = profile_dataset(train_subset, test_features)
            selector_result = select_model(
                selector_profile,
                X_train,
                y_train,
                stock_number=stock_number,
                verbose=False,
            )
            selector_prediction = float(selector_result["model"].predict(test_features)[0])

            baseline_model = _build_linear_baseline()
            baseline_model.fit(X_train, y_train)
            baseline_prediction = float(baseline_model.predict(test_features)[0])
        except Exception as exc:
            failed_trials.append({"trial": int(trial_idx), "error": str(exc)})
            if verbose:
                print(f"  Trial {trial_idx} failed: {exc}")
            continue

        selector_error = selector_prediction - true_price
        baseline_error = baseline_prediction - true_price
        selector_abs_error = abs(selector_error)
        baseline_abs_error = abs(baseline_error)

        if selector_abs_error + 1e-12 < baseline_abs_error:
            winner = "selector"
        elif baseline_abs_error + 1e-12 < selector_abs_error:
            winner = "linear_baseline"
        else:
            winner = "tie"

        trial_results.append({
            "trial": int(trial_idx),
            "true_price": true_price,
            "selector_prediction": selector_prediction,
            "baseline_prediction": baseline_prediction,
            "selector_error": selector_error,
            "baseline_error": baseline_error,
            "selector_abs_error": selector_abs_error,
            "baseline_abs_error": baseline_abs_error,
            "winner": winner,
        })

    if not trial_results:
        return {
            "stock_number": stock_number,
            "n_trials": 0,
            "failed_trials": failed_trials,
            "selector_beats_baseline": False,
            "message": "No comparison trials completed successfully.",
            "saved_model_path": None,
        }

    trial_df = pd.DataFrame(trial_results)
    selector_metrics = _summarize_prediction_errors(trial_df["selector_error"].values)
    baseline_metrics = _summarize_prediction_errors(trial_df["baseline_error"].values)

    selector_wins = int((trial_df["winner"] == "selector").sum())
    baseline_wins = int((trial_df["winner"] == "linear_baseline").sum())
    ties = int((trial_df["winner"] == "tie").sum())
    selector_beats_baseline = selector_metrics["rmse"] < baseline_metrics["rmse"]

    saved_model_path = None
    final_selector_result = None
    final_selector_prediction = None

    if selector_beats_baseline:
        profile = profile_dataset(train_df, test_df)
        X_full = train_df.drop(columns=["target"])
        y_full = train_df["target"]
        final_selector_result = select_model(
            profile,
            X_full,
            y_full,
            stock_number=stock_number,
            verbose=verbose,
        )
        final_predictions = final_selector_result["model"].predict(test_df)
        final_selector_prediction = final_predictions.tolist()
        saved_model_path = save_selector_model_artifact(
            stock_number=stock_number,
            train_df=train_df,
            test_df=test_df,
            selector_result=final_selector_result,
            save_dir=save_dir,
        )

    result = {
        "stock_number": stock_number,
        "n_trials": int(len(trial_df)),
        "selector": selector_metrics,
        "linear_baseline": baseline_metrics,
        "selector_wins": selector_wins,
        "baseline_wins": baseline_wins,
        "ties": ties,
        "selector_win_rate_pct": float(100.0 * selector_wins / len(trial_df)),
        "rmse_improvement": float(baseline_metrics["rmse"] - selector_metrics["rmse"]),
        "mae_improvement": float(baseline_metrics["mae"] - selector_metrics["mae"]),
        "selector_beats_baseline": selector_beats_baseline,
        "failed_trials": failed_trials,
        "saved_model_path": str(saved_model_path) if saved_model_path is not None else None,
        "final_selector_prediction": final_selector_prediction,
        "final_selector_summary": _model_package(final_selector_result) if final_selector_result is not None else None,
    }

    if verbose:
        print(f"\n  Selector RMSE:        {selector_metrics['rmse']:.4f}")
        print(f"  Linear baseline RMSE: {baseline_metrics['rmse']:.4f}")
        print(f"  Selector MAE:         {selector_metrics['mae']:.4f}")
        print(f"  Linear baseline MAE:  {baseline_metrics['mae']:.4f}")
        print(f"  Trial wins:           selector {selector_wins} | baseline {baseline_wins} | ties {ties}")
        if selector_beats_baseline:
            print(f"  Verdict: selector beat the baseline and was saved to {saved_model_path}")
        else:
            print("  Verdict: selector did not beat the basic linear regression baseline, so no model was saved.")

    return result


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
        
        metrics = backtest_stock(train, stock_number=i, n_trials=n_trials_per_stock, verbose=False)
        
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
    parser.add_argument("--stock", type=int, default=None, help="Evaluate one stock against a basic linear regression baseline")
    parser.add_argument("--save_dir", type=str, default=None, help="Where to save winning selector models")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for evaluation sampling")
    args = parser.parse_args()

    if args.stock is not None:
        evaluate_stock_selector_vs_linear(
            stock_number=args.stock,
            data_dir=args.data_dir,
            n_trials=args.trials,
            seed=args.seed,
            save_dir=args.save_dir,
            verbose=True,
        )
    else:
        data_dir = _resolve_data_dir(args.data_dir)
        run_full_backtest(data_dir, n_trials_per_stock=args.trials)
