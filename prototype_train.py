from dataclasses import asdict
from pathlib import Path

import pandas as pd

from pipeline.data_check import profile_dataset
from pipeline.model_select import select_model

DATA_DIR = Path(__file__).resolve().parent.parent / "hackathon_data"


def get_data_path(stock_number, split="train", data_dir=DATA_DIR):
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    return Path(data_dir) / f"stock_{stock_number}_{split}.csv"


def load_data(stock_number, data_dir=DATA_DIR):
    train_path = get_data_path(stock_number, "train", data_dir)
    test_path = get_data_path(stock_number, "test", data_dir)

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found at {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Testing data not found at {test_path}")

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data


def build_stock_context(stock_number, data_dir=DATA_DIR):
    train_data, test_data = load_data(stock_number, data_dir=data_dir)
    profile = profile_dataset(train_data, test_data)
    X_train = train_data.drop(columns=["target"])
    y_train = train_data["target"]

    return {
        "stock_number": stock_number,
        "train_data": train_data,
        "test_data": test_data,
        "X_train": X_train,
        "y_train": y_train,
        "profile": profile,
    }


def analyze_stock(stock_number, data_dir=DATA_DIR, verbose=False):
    context = build_stock_context(stock_number, data_dir=data_dir)
    selection = select_model(
        context["profile"],
        context["X_train"],
        context["y_train"],
        stock_number=stock_number,
        verbose=verbose,
    )
    test_prediction = float(selection["model"].predict(context["test_data"])[0])

    return {
        "stock_number": stock_number,
        "profile": asdict(context["profile"]),
        "best_model": selection["name"],
        "best_params": selection["best_params"],
        "cv_rmse": selection["cv_rmse"],
        "cv_std": selection["cv_std"],
        "train_rmse": selection["train_rmse"],
        "overfit_ratio": selection["overfit_ratio"],
        "overfit_flag": selection["overfit_flag"],
        "tie_tolerance": selection["tie_tolerance"],
        "all_scores": selection["all_scores"],
        "tuned_models": selection["tuned_models"],
        "test_prediction": test_prediction,
        "model": selection["model"],
    }


def run_all_stocks(stock_numbers=None, data_dir=DATA_DIR, verbose=False):
    if stock_numbers is None:
        stock_numbers = range(1, 10)

    return [
        analyze_stock(stock_number, data_dir=data_dir, verbose=verbose)
        for stock_number in stock_numbers
    ]


def main(stock_numbers=None, data_dir=DATA_DIR, verbose=False):
    return run_all_stocks(stock_numbers=stock_numbers, data_dir=data_dir, verbose=verbose)


if __name__ == "__main__":
    main(verbose=False)
