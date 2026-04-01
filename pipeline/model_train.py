"""
Train and save the selected model for a single stock.
"""

import pickle
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).parent))
from data_check import profile_dataset
from model_select import DEFAULT_RANDOM_STATE, select_model


DEFAULT_DATA_DIR = Path(__file__).parent.parent / "hackathon_data"
DEFAULT_SAVED_MODELS_DIR = Path(__file__).parent.parent / "saved_models"

# Resolve the data directory and ensure it exists
def _resolve_data_dir(data_dir=None):
    return Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR

# Resolve the save directory and ensure it exists
def _resolve_save_dir(save_dir=None):
    resolved = Path(save_dir) if save_dir is not None else DEFAULT_SAVED_MODELS_DIR
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved

# Return a DataFrame with only the feature columns (dropping target if present)
def _feature_frame(df: pd.DataFrame, target_col="target") -> pd.DataFrame:
    return df.drop(columns=[target_col]) if target_col in df.columns else df.copy()

# Extract a summary of the selector result without the model object (for easier printing and saving metadata)
def _selector_summary(result):
    return {key: value for key, value in result.items() if key != "model"}

# Sort a score map (dict of model name to score) into a list of tuples sorted by score
def _sorted_scores(score_map):
    return sorted(score_map.items(), key=lambda item: item[1])

# Save the trained model and metadata for a single stock(winner of model selection)
def save_trained_stock_model(
    stock_number,
    train_df,
    test_df,
    profile,
    selector_result,
    save_dir=None,
    target_col="target",
):
    """
    Save the fitted winning model and the metadata needed to reuse it later.
    """
    #Resolve directories and prepare data for prediction
    save_dir = _resolve_save_dir(save_dir)
    test_features = _feature_frame(test_df, target_col=target_col)

    # Get the test prediction from the winning model
    test_prediction = selector_result["model"].predict(test_features)

    # Create an artifact dictionary containing all relevant information about the training outcome
    artifact = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_version": 1,
        "stock_number": int(stock_number),
        "target_column": target_col,
        "feature_columns": _feature_frame(train_df, target_col=target_col).columns.tolist(),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "profile": asdict(profile),
        "model": selector_result["model"],
        "model_summary": _selector_summary(selector_result),
        "test_prediction": test_prediction.tolist(),
    }

    #Create path and save the artifact as a pickle file
    artifact_path = save_dir / f"stock_{int(stock_number)}_winner.pkl"
    with open(artifact_path, "wb") as handle:
        pickle.dump(artifact, handle)
    return artifact_path

# Load a saved model artifact for a stock and return the model object and metadata
def load_trained_stock_model(model_path):
    with open(model_path, "rb") as handle:
        return pickle.load(handle)

# Print a summary of the training results for a stock in a readable format
def _print_training_summary(stock_number, profile, selector_result, artifact_path, test_prediction):
    all_scores = _sorted_scores(selector_result["all_scores"])
    mixing_scores = _sorted_scores(selector_result["mixing_scores"])

    print("\n" + "=" * 90)
    print(f"  STOCK {stock_number} TRAINING SUMMARY")
    print("=" * 90)
    print(
        f"  Data: rows={profile.n_train}, features={profile.n_features}, "
        f"signal={profile.signal_strength}, collinearity={profile.high_collinearity}, "
        f"heavy_tails={profile.heavy_tails}, test_ood={profile.test_ood}"
    )
    print(
        f"  Winner: {selector_result['name']}  |  CV RMSE={selector_result['cv_rmse']:.4f}  "
        f"+/- {selector_result['cv_std']:.4f}"
    )
    print(
        f"  Train RMSE={selector_result['train_rmse']:.4f}  |  "
        f"Overfit ratio={selector_result['overfit_ratio']:.4f}  |  "
        f"Noise ratio={selector_result['noise_ratio']:.4f}"
    )
    print(f"  Best params: {selector_result['best_params']}")

    if selector_result["mixing_used"]:
        print(f"  Winner type: ensemble ({selector_result['mixing_method']})")
        print(f"  Ensemble members: {selector_result['mixing_members']}")
        if selector_result["mixing_weights"] is not None:
            print(f"  Ensemble weights: {selector_result['mixing_weights']}")
    else:
        print(f"  Winner type: single model ({selector_result['best_single_model']})")

    print("\n  Top candidate models:")
    for name, rmse in all_scores[:5]:
        print(f"    {name:<24} RMSE={rmse:.4f}")

    if mixing_scores:
        print("\n  Top mixed models:")
        for name, rmse in mixing_scores[:5]:
            print(f"    {name:<24} RMSE={rmse:.4f}")

    print(f"\n  Test prediction(s): {test_prediction}")
    print(f"  Saved model artifact: {artifact_path}")

# Run the selection and training process for a single stock, print the summary, and save the winning model artifact.
def train_single_stock(
    stock_number,
    data_dir=None,
    save_dir=None,
    verbose=True,
    random_state=DEFAULT_RANDOM_STATE,
    target_col="target",
):
    """
    Train the selector on one stock, print the training outcome, and save the winner.
    """
    # Validate stock number
    stock_number = int(stock_number)
    if stock_number < 1 or stock_number > 9:
        raise ValueError("stock_number must be between 1 and 9")

    # Resolve data directory
    data_dir = _resolve_data_dir(data_dir)

    # Load the train and test data for the specified stock
    train_path = data_dir / f"stock_{stock_number}_train.csv"
    test_path = data_dir / f"stock_{stock_number}_test.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Split into features and target
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    test_features = _feature_frame(test_df, target_col=target_col)

    # Returns a profile for the stock.
    profile = profile_dataset(train_df, test_features, target_col=target_col)

    # Run the model selection process to find the best model for this stock based on the profile and training data.
    selector_result = select_model(
        profile,
        X_train,
        y_train,
        stock_number=stock_number,
        verbose=verbose,
        random_state=random_state,
    )

    # Predict test results using the winning model
    test_prediction = selector_result["model"].predict(test_features).tolist()
    
    # Save the winning model and relevant metadata as a pickle file for later use.
    artifact_path = save_trained_stock_model(
        stock_number=stock_number,
        train_df=train_df,
        test_df=test_df,
        profile=profile,
        selector_result=selector_result,
        save_dir=save_dir,
        target_col=target_col,
    )

    # Print a detailed summary of the training results, including data profile, model performance, and test predictions.
    if verbose:
        _print_training_summary(
            stock_number=stock_number,
            profile=profile,
            selector_result=selector_result,
            artifact_path=artifact_path,
            test_prediction=test_prediction,
        )

    # Return a dictionary with the key information about the training outcome, including paths, predictions, profile, and model summary (excluding the model object itself for easier serialization).
    return {
        "stock_number": stock_number,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "artifact_path": str(artifact_path),
        "test_prediction": test_prediction,
        "profile": asdict(profile),
        "result": selector_result,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and save the best selector model for one stock")
    parser.add_argument("--stock", type=int, required=True, help="Stock number to train (1-9)")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to hackathon_data folder")
    parser.add_argument("--save_dir", type=str, default=None, help="Where to save the trained model artifact")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE, help="Random seed for model selection")
    args = parser.parse_args()

    train_single_stock(
        stock_number=args.stock,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        verbose=True,
        random_state=args.seed,
    )
