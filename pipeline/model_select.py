import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor, VotingRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

SCORING = "neg_root_mean_squared_error"
DEFAULT_RANDOM_STATE = 2003

# Function to build cross-validation strategy based on number of training samples
def _build_cv(n_rows, random_state):
    if n_rows < 40:
        n_splits = 3
    elif n_rows < 120:
        n_splits = 4
    else:
        n_splits = 5

    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

# Function to build candidate models and their hyperparameter grids based on stock profile
def _build_candidate_spaces(profile, random_state):
    candidate_spaces = {
        "LinearRegression": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("linear", LinearRegression()),
            ]),
            "param_grid": {},
        },
        "Ridge": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge()),
            ]),
            "param_grid": {
                "ridge__alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            },
        },
        "Lasso": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("lasso", Lasso(max_iter=10000, random_state=random_state)),
            ]),
            "param_grid": {
                "lasso__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
            },
        },
        "BayesianRidge": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("bayes", BayesianRidge()),
            ]),
            "param_grid": {
                "bayes__alpha_1": [1e-6, 1e-5],
                "bayes__lambda_1": [1e-6, 1e-5],
            },
        },
        "ElasticNet": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("enet", ElasticNet(max_iter=10000, random_state=random_state)),
            ]),
            "param_grid": {
                "enet__alpha": [0.001, 0.01, 0.1, 1.0],
                "enet__l1_ratio": [0.2, 0.5, 0.8],
            },
        },
    }

    if profile.try_poly_features:
        candidate_spaces["Ridge+Poly"] = {
            "estimator": Pipeline([
                ("poly", PolynomialFeatures(include_bias=False)),
                ("scaler", StandardScaler()),
                ("ridge", Ridge()),
            ]),
            "param_grid": {
                "poly__degree": [2, 3] if profile.n_train >= 150 else [2],
                "ridge__alpha": [0.01, 0.1, 1, 10, 100],
            },
        }

    if profile.n_train <= 500:
        candidate_spaces["KNN"] = {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("knn", KNeighborsRegressor()),
            ]),
            "param_grid": {
                "knn__n_neighbors": [3, 5, 7, 11],
                "knn__weights": ["uniform", "distance"],
            },
        }

        candidate_spaces["SVR"] = {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("svr", SVR()),
            ]),
            "param_grid": {
                "svr__kernel": ["rbf", "linear"],
                "svr__C": [0.1, 1, 10],
                "svr__epsilon": [0.01, 0.1, 0.5],
            },
        }

    if profile.n_train >= 80:
        candidate_spaces["GradientBoosting"] = {
            "estimator": GradientBoostingRegressor(random_state=random_state),
            "param_grid": {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [2, 3, 4],
                "loss": ["huber", "squared_error"] if profile.use_robust_loss else ["squared_error"],
            },
        }

        candidate_spaces["LightGBM"] = {
            "estimator": lgb.LGBMRegressor(
                random_state=random_state,
                verbose=-1,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=1,
            ),
            "param_grid": {
                "n_estimators": [100, 200],
                "num_leaves": [15, 31],
                "learning_rate": [0.05, 0.1],
            },
        }

        candidate_spaces["XGBoost"] = {
            "estimator": XGBRegressor(
                random_state=random_state,
                verbosity=0,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=1,
            ),
            "param_grid": {
                "n_estimators": [100, 200],
                "max_depth": [3, 4],
                "learning_rate": [0.05, 0.1],
            },
        }

    return candidate_spaces

# Function to tune a single candidate model and return its best estimator and scores
def tune_candidate_model(name, estimator, param_grid, X_train, y_train, cv):
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=SCORING,
        cv=cv,
        refit=True,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    best_index = search.best_index_
    best_rmse = float(-search.best_score_)
    best_std = float(search.cv_results_["std_test_score"][best_index])

    return {
        "name": name,
        "best_estimator": search.best_estimator_,
        "best_params": search.best_params_,
        "tune_rmse": best_rmse,
        "tune_std": best_std,
    }

# Function to tune all candidate models and return their best estimators and scores
def tune_all_candidates(profile, X_train, y_train, verbose=True, random_state=DEFAULT_RANDOM_STATE):
    candidate_spaces = _build_candidate_spaces(profile, random_state=random_state)
    tuning_cv = _build_cv(profile.n_train, random_state=random_state)
    tuned_candidates = {}

    if verbose:
        print(f"\n  {'Model':<20} {'Tune RMSE':>10}  {'+/- std':>8}")
        print(f"  {'-' * 44}")

    for name, config in candidate_spaces.items():
        tuned = tune_candidate_model(
            name,
            config["estimator"],
            config["param_grid"],
            X_train,
            y_train,
            tuning_cv,
        )
        tuned_candidates[name] = tuned

        if verbose:
            print(f"  {name:<20} {tuned['tune_rmse']:>10.3f}  {tuned['tune_std']:>8.3f}")

    return tuned_candidates

# Function to compare tuned candidates using cross-validation on the training set
def compare_tuned_candidates(tuned_candidates, X_train, y_train, cv, verbose=True):
    comparison = {}

    if verbose:
        print(f"\n  {'Model':<20} {'Final RMSE':>10}  {'+/- std':>8}")
        print(f"  {'-' * 45}")

    for name, tuned in tuned_candidates.items():
        scores = cross_val_score(
            tuned["best_estimator"],
            X_train,
            y_train,
            scoring=SCORING,
            cv=cv,
        )
        rmse = float(-scores.mean())
        std = float(scores.std())
        comparison[name] = {
            "cv_rmse": rmse,
            "cv_std": std,
        }

        if verbose:
            print(f"  {name:<20} {rmse:>10.3f}  {std:>8.3f}")

    best_name = min(comparison, key=lambda model_name: comparison[model_name]["cv_rmse"])
    return best_name, comparison[best_name], comparison

# Function to create and evaluate ensemble models from top candidates
def mix_models(
    ordered_candidates,
    X_train,
    y_train,
    cv,
    verbose=True,
    top_k=3,
    random_state=DEFAULT_RANDOM_STATE,
):
    ensemble_comparison = {}

    if len(ordered_candidates) < 2:
        return None, ensemble_comparison

    max_models = min(top_k, len(ordered_candidates))
    stacking_cv = _build_cv(len(X_train), random_state=random_state + 2)

    if verbose:
        print(f"\n  {'Ensemble':<32} {'RMSE':>10}  {'+/- std':>8}")
        print(f"  {'-' * 53}")

    for n_models in range(2, max_models + 1):
        members = ordered_candidates[:n_models]
        member_names = [candidate["name"] for candidate in members]
        estimators = [
            (candidate["name"], clone(candidate["best_estimator"]))
            for candidate in members
        ]

        rmse_values = np.array([candidate["cv_rmse"] for candidate in members], dtype=float)
        safe_rmse = np.maximum(rmse_values, 1e-9)
        weights = (1.0 / safe_rmse).tolist()

        ensemble_specs = [
            {
                "name": f"SimpleAverage(top{n_models})",
                "method": "simple_average",
                "members": member_names,
                "estimator": VotingRegressor(
                    estimators=estimators,
                    weights=[1.0] * n_models,
                ),
            },
            {
                "name": f"WeightedAverage(top{n_models})",
                "method": "weighted_average",
                "members": member_names,
                "weights": weights,
                "estimator": VotingRegressor(
                    estimators=estimators,
                    weights=weights,
                ),
            },
            {
                "name": f"Stacking(top{n_models})",
                "method": "stacking",
                "members": member_names,
                "estimator": StackingRegressor(
                    estimators=estimators,
                    final_estimator=Ridge(alpha=1.0),
                    cv=stacking_cv,
                    n_jobs=-1,
                ),
            },
        ]

        for spec in ensemble_specs:
            scores = cross_val_score(
                spec["estimator"],
                X_train,
                y_train,
                scoring=SCORING,
                cv=cv,
            )
            rmse = float(-scores.mean())
            std = float(scores.std())

            ensemble_comparison[spec["name"]] = {
                "cv_rmse": rmse,
                "cv_std": std,
                "method": spec["method"],
                "members": spec["members"],
                "weights": spec.get("weights"),
                "estimator": spec["estimator"],
            }

            if verbose:
                print(f"  {spec['name']:<32} {rmse:>10.3f}  {std:>8.3f}")

    if not ensemble_comparison:
        return None, ensemble_comparison

    best_name = min(
        ensemble_comparison,
        key=lambda model_name: ensemble_comparison[model_name]["cv_rmse"],
    )
    best_result = ensemble_comparison[best_name].copy()
    best_result["name"] = best_name
    return best_result, ensemble_comparison

# Main function to select the best model (or ensemble) based on stock profile
def select_model(profile, X_train, y_train, verbose=True, random_state=DEFAULT_RANDOM_STATE):
    tuned_candidates = tune_all_candidates(
        profile,
        X_train,
        y_train,
        verbose=verbose,
        random_state=random_state,
    )
    comparison_cv = _build_cv(profile.n_train, random_state=random_state + 1)
    best_name, best_scores, comparison = compare_tuned_candidates(
        tuned_candidates,
        X_train,
        y_train,
        comparison_cv,
        verbose=verbose,
    )

    ordered_candidates = []
    for model_name, scores in sorted(comparison.items(), key=lambda kv: kv[1]["cv_rmse"]):
        tuned = tuned_candidates[model_name]
        ordered_candidates.append({
            "name": model_name,
            "best_estimator": tuned["best_estimator"],
            "best_params": tuned["best_params"],
            "tune_rmse": tuned["tune_rmse"],
            "tune_std": tuned["tune_std"],
            "cv_rmse": scores["cv_rmse"],
            "cv_std": scores["cv_std"],
        })

    mix_winner, ensemble_comparison = mix_models(
        ordered_candidates,
        X_train,
        y_train,
        comparison_cv,
        verbose=verbose,
        random_state=random_state,
    )

    final_name = best_name
    final_model = clone(tuned_candidates[best_name]["best_estimator"])
    final_best_params = tuned_candidates[best_name]["best_params"]
    final_rmse = best_scores["cv_rmse"]
    final_std = best_scores["cv_std"]
    mixing_used = False
    mixing_method = None
    mixing_members = []
    mixing_weights = None

    if mix_winner and mix_winner["cv_rmse"] < final_rmse:
        final_name = mix_winner["name"]
        final_model = clone(mix_winner["estimator"])
        final_best_params = {
            "ensemble_method": mix_winner["method"],
            "members": mix_winner["members"],
        }
        if mix_winner.get("weights") is not None:
            final_best_params["weights"] = mix_winner["weights"]
        final_rmse = mix_winner["cv_rmse"]
        final_std = mix_winner["cv_std"]
        mixing_used = True
        mixing_method = mix_winner["method"]
        mixing_members = mix_winner["members"]
        mixing_weights = mix_winner.get("weights")

    final_model.fit(X_train, y_train)

    if verbose:
        if mixing_used:
            print(f"\n  Winner: {final_name}  (RMSE={final_rmse:.3f} +/- {final_std:.3f})")
            print(f"  Ensemble beat best single model {best_name} ({best_scores['cv_rmse']:.3f})")
        else:
            print(f"\n  Winner: {best_name}  (RMSE={final_rmse:.3f} +/- {final_std:.3f})")

    noise_ratio = final_rmse / profile.target_std if profile.target_std > 0 else 1.0
    profile.noise_ratio = noise_ratio
    profile.no_quote = noise_ratio > 0.98

    tuned_summary = {}
    for name, tuned in tuned_candidates.items():
        tuned_summary[name] = {
            "best_params": tuned["best_params"],
            "tune_rmse": tuned["tune_rmse"],
            "tune_std": tuned["tune_std"],
            "final_rmse": comparison[name]["cv_rmse"],
            "final_std": comparison[name]["cv_std"],
        }

    return {
        "name": final_name,
        "model": final_model,
        "best_params": final_best_params,
        "cv_rmse": final_rmse,
        "cv_std": final_std,
        "all_scores": {name: scores["cv_rmse"] for name, scores in comparison.items()},
        "mixing_scores": {name: scores["cv_rmse"] for name, scores in ensemble_comparison.items()},
        "tuned_models": tuned_summary,
        "noise_ratio": noise_ratio,
        "blended": mixing_used,
        "mixing_used": mixing_used,
        "mixing_method": mixing_method,
        "mixing_members": mixing_members,
        "mixing_weights": mixing_weights,
        "best_single_model": best_name,
        "best_single_rmse": best_scores["cv_rmse"],
        "random_state": random_state,
    }


if __name__ == "__main__":
    from pathlib import Path
    import sys

    sys.path.append(str(Path(__file__).parent))
    from data_check import profile_dataset

    DATA_DIR = Path(__file__).parent.parent / "hackathon_data"

    for i in range(1, 10):
        print(f"\n{'=' * 50}")
        print(f"  STOCK {i}")
        print(f"{'=' * 50}")
        train = pd.read_csv(DATA_DIR / f"stock_{i}_train.csv")
        test = pd.read_csv(DATA_DIR / f"stock_{i}_test.csv")
        X = train.drop("target", axis=1)
        y = train["target"]
        profile = profile_dataset(train, test)
        result = select_model(profile, X, y)
        pred = result["model"].predict(test)[0]
        print(f"  Prediction:   {pred:.2f}")
        print(f"  Noise ratio:  {result['noise_ratio']:.4f}")
        print(f"  Best params:  {result['best_params']}")
