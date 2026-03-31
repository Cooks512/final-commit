import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

from sklearn.utils import resample

warnings.filterwarnings("ignore")

SCORING = "neg_root_mean_squared_error"


# Returns the best CV for given model and params.
def _build_cv(n_rows, random_state):
    if n_rows < 40:
        n_splits = 3
    elif n_rows < 120:
        n_splits = 4
    else:
        n_splits = 5

    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


# Builds candidate models and their tuning spaces based on dataset profile
def _build_candidate_spaces(profile):
    candidate_spaces = {
        "LinearRegression": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("linear", LinearRegression())
            ]),
            "param_grid": {},
        },
        "Ridge": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge())
            ]),
            "param_grid": {
                "ridge__alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            },
        },
        "Lasso": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("lasso", Lasso(max_iter=10000, random_state=42))
            ]),
            "param_grid": {
                "lasso__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
            },
        },
        "BayesianRidge": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("bayes", BayesianRidge())
            ]),
            "param_grid": {
                "bayes__alpha_1": [1e-6, 1e-5],
                "bayes__lambda_1": [1e-6, 1e-5],
            },
        },
        "ElasticNet": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("enet", ElasticNet(max_iter=10000, random_state=42))
            ]),
            "param_grid": {
                "enet__alpha": [0.001, 0.01, 0.1, 1.0],
                "enet__l1_ratio": [0.2, 0.5, 0.8],
            },
        },
    }

    # Add polynomial features if profile recommends
    if profile.try_poly_features:
        candidate_spaces["Ridge+Poly"] = {
            "estimator": Pipeline([
                ("poly", PolynomialFeatures(include_bias=False)),
                ("scaler", StandardScaler()),
                ("ridge", Ridge())
            ]),
            "param_grid": {
                "poly__degree": [2, 3] if profile.n_train >= 150 else [2],
                "ridge__alpha": [0.01, 0.1, 1, 10, 100],
            },
        }

    # KNN — best chance on small datasets (stocks 3, 9) where nearest-neighbour
    # can beat linear models. Gated to n_train <= 500 to avoid slow fits on large data.
    # SVR — support vector regression, also gated to smaller datasets.
    if profile.n_train <= 500:
        candidate_spaces["KNN"] = {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("knn", KNeighborsRegressor())
            ]),
            "param_grid": {
                "knn__n_neighbors": [3, 5, 7, 11],
                "knn__weights": ["uniform", "distance"],
            },
        }

        candidate_spaces["SVR"] = {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("svr", SVR())
            ]),
            "param_grid": {
                "svr__kernel": ["rbf", "linear"],
                "svr__C": [0.1, 1, 10],
                "svr__epsilon": [0.01, 0.1, 0.5],
            },
        }

    # Add tree-based models for larger datasets
    if profile.n_train >= 80:
        # GradientBoosting — uses Huber loss when heavy_tails detected,
        # which downweights outlier targets. Ties into use_robust_loss from data_check.
        candidate_spaces["GradientBoosting"] = {
            "estimator": GradientBoostingRegressor(
                random_state=42,
            ),
            "param_grid": {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [2, 3, 4],
                "loss": ["huber", "squared_error"] if profile.use_robust_loss else ["squared_error"],
            },
        }

        candidate_spaces["LightGBM"] = {
            "estimator": lgb.LGBMRegressor(
                random_state=42,
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
                random_state=42,
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


# Tune parameters for a single candidate model
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


# Tunes all candidate models for a stock
def tune_all_candidates(profile, X_train, y_train, verbose=True):
    candidate_spaces = _build_candidate_spaces(profile)
    tuning_cv = _build_cv(profile.n_train, random_state=42)
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


# Compares the tuned candidate models using a fresh CV split
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


# Select the best model for a stock
#def select_model(profile, X_train, y_train, verbose=True):
#
#    tuned_candidates = tune_all_candidates(profile, X_train, y_train, verbose=verbose)
#    comparison_cv = _build_cv(profile.n_train, random_state=99)
#    best_name, best_scores, comparison = compare_tuned_candidates(
#        tuned_candidates,
#        X_train,
#        y_train,
#        comparison_cv,
#        verbose=verbose,
#    )
#
#    final_model = clone(tuned_candidates[best_name]["best_estimator"])
#    final_model.fit(X_train, y_train)
#
#    best_rmse = best_scores["cv_rmse"]
#    best_std = best_scores["cv_std"]
#    noise_ratio = best_rmse / profile.target_std if profile.target_std > 0 else 1.0
#
#    # --- Blend top-2 if they're close and signal is strong ---
#    sorted_models = sorted(comparison.items(), key=lambda kv: kv[1]["cv_rmse"])
#    first_name,  first_scores  = sorted_models[0]
#    second_name, second_scores = sorted_models[1]
#    first_rmse  = first_scores["cv_rmse"]
#    second_rmse = second_scores["cv_rmse"]
#    gap_pct = (second_rmse - first_rmse) / first_rmse * 100 if first_rmse > 0 else 100
#
#    use_blend = gap_pct < 1.5 and noise_ratio < 0.7
#
#    if use_blend:
#        m1 = clone(tuned_candidates[first_name]["best_estimator"])
#        m2 = clone(tuned_candidates[second_name]["best_estimator"])
#        m1.fit(X_train, y_train)
#        m2.fit(X_train, y_train)
#
#        w1 = 1.0 / first_rmse
#        w2 = 1.0 / second_rmse
#        w_total = w1 + w2
#        w1, w2 = w1 / w_total, w2 / w_total
#
#        class BlendModel:
#            def __init__(self, m1, m2, w1, w2):
#                self.m1, self.m2, self.w1, self.w2 = m1, m2, w1, w2
#            def predict(self, X):
#                return self.w1 * self.m1.predict(X) + self.w2 * self.m2.predict(X)
#            def fit(self, X, y):
#                return self
#
#        final_model = BlendModel(m1, m2, w1, w2)
#        best_name = f"Blend({first_name}+{second_name})"
#        best_rmse = first_rmse  # conservative
#
#        if verbose:
#            print(f"\n  Top-2 within {gap_pct:.1f}% — blending {first_name} ({w1:.2f}) + {second_name} ({w2:.2f})")
#            print(f"  Winner: {best_name}  (RMSE≈{best_rmse:.3f})")
#    else:
#        if verbose:
#            print(f"\n  Winner: {best_name}  (RMSE={best_rmse:.3f} +/- {best_std:.3f})")
#
#    profile.noise_ratio = noise_ratio
#    profile.no_quote = noise_ratio > 0.98
#
#    tuned_summary = {}
#    for name, tuned in tuned_candidates.items():
#        tuned_summary[name] = {
#            "best_params": tuned["best_params"],
#            "tune_rmse": tuned["tune_rmse"],
#            "tune_std": tuned["tune_std"],
#            "final_rmse": comparison[name]["cv_rmse"],
#            "final_std": comparison[name]["cv_std"],
#        }
#
#    # When blended, best_params comes from the top model
#    if use_blend:
#        best_params = tuned_candidates[first_name]["best_params"]
#    else:
#        best_params = tuned_candidates[best_name]["best_params"]
#
#    return {
#        "name": best_name,
#        "model": final_model,
#        "best_params": best_params,
#        "cv_rmse": best_rmse,
#        "cv_std": best_std,
#        "all_scores": {name: scores["cv_rmse"] for name, scores in comparison.items()},
#        "tuned_models": tuned_summary,
#        "noise_ratio": noise_ratio,
#        "blended": use_blend,
#    }
#

def select_model(profile, X_train, y_train, verbose=True):

    # Augment only small datasets
    if profile.n_train < 200:
        n_bootstrap = 8 if profile.n_train < 50 else 4

        X_aug_parts = [X_train.copy()]
        y_aug_parts = [y_train.copy()]

        y_noise_std = max(profile.target_std * 0.02, 1e-8)

        for _ in range(n_bootstrap):
            X_boot, y_boot = resample(X_train, y_train, replace=True, random_state=None)

            X_boot = X_boot.copy()
            y_boot = y_boot.copy()

            x_noise = np.random.normal(0, 0.01, size=X_boot.shape)
            y_noise = np.random.normal(0, y_noise_std, size=len(y_boot))

            if isinstance(X_boot, pd.DataFrame):
                X_boot = X_boot + x_noise
            else:
                X_boot = X_boot + x_noise

            if isinstance(y_boot, pd.Series):
                y_boot = y_boot + y_noise
            else:
                y_boot = y_boot + y_noise

            X_aug_parts.append(X_boot)
            y_aug_parts.append(y_boot)

        if isinstance(X_train, pd.DataFrame):
            X_train = pd.concat(X_aug_parts, axis=0, ignore_index=True)
        else:
            X_train = np.vstack(X_aug_parts)

        if isinstance(y_train, pd.Series):
            y_train = pd.concat(y_aug_parts, axis=0, ignore_index=True)
        else:
            y_train = np.hstack(y_aug_parts)

        if verbose:
            print(f"\n  Augmented training rows: {profile.n_train} -> {len(y_train)}")

    tuned_candidates = tune_all_candidates(profile, X_train, y_train, verbose=verbose)
    comparison_cv = _build_cv(profile.n_train, random_state=99)
    best_name, best_scores, comparison = compare_tuned_candidates(
        tuned_candidates,
        X_train,
        y_train,
        comparison_cv,
        verbose=verbose,
    )

    final_model = clone(tuned_candidates[best_name]["best_estimator"])
    final_model.fit(X_train, y_train)

    best_rmse = best_scores["cv_rmse"]
    best_std = best_scores["cv_std"]
    noise_ratio = best_rmse / profile.target_std if profile.target_std > 0 else 1.0

    sorted_models = sorted(comparison.items(), key=lambda kv: kv[1]["cv_rmse"])
    first_name,  first_scores  = sorted_models[0]
    second_name, second_scores = sorted_models[1]
    first_rmse  = first_scores["cv_rmse"]
    second_rmse = second_scores["cv_rmse"]
    gap_pct = (second_rmse - first_rmse) / first_rmse * 100 if first_rmse > 0 else 100

    use_blend = gap_pct < 1.5 and noise_ratio < 0.7

    if use_blend:
        m1 = clone(tuned_candidates[first_name]["best_estimator"])
        m2 = clone(tuned_candidates[second_name]["best_estimator"])
        m1.fit(X_train, y_train)
        m2.fit(X_train, y_train)

        w1 = 1.0 / first_rmse
        w2 = 1.0 / second_rmse
        w_total = w1 + w2
        w1, w2 = w1 / w_total, w2 / w_total

        class BlendModel:
            def __init__(self, m1, m2, w1, w2):
                self.m1, self.m2, self.w1, self.w2 = m1, m2, w1, w2
            def predict(self, X):
                return self.w1 * self.m1.predict(X) + self.w2 * self.m2.predict(X)
            def fit(self, X, y):
                return self

        final_model = BlendModel(m1, m2, w1, w2)
        best_name = f"Blend({first_name}+{second_name})"
        best_rmse = first_rmse

        if verbose:
            print(f"\n  Top-2 within {gap_pct:.1f}% — blending {first_name} ({w1:.2f}) + {second_name} ({w2:.2f})")
            print(f"  Winner: {best_name}  (RMSE≈{best_rmse:.3f})")
    else:
        if verbose:
            print(f"\n  Winner: {best_name}  (RMSE={best_rmse:.3f} +/- {best_std:.3f})")

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

    if use_blend:
        best_params = tuned_candidates[first_name]["best_params"]
    else:
        best_params = tuned_candidates[best_name]["best_params"]

    return {
        "name": best_name,
        "model": final_model,
        "best_params": best_params,
        "cv_rmse": best_rmse,
        "cv_std": best_std,
        "all_scores": {name: scores["cv_rmse"] for name, scores in comparison.items()},
        "tuned_models": tuned_summary,
        "noise_ratio": noise_ratio,
        "blended": use_blend,
    }


# Test
if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent))
    from data_check import profile_dataset

    DATA_DIR = Path(__file__).parent.parent / "hackathon_data"

    for i in range(1, 10):

        print(f"\n{'='*50}")
        print(f"  STOCK {i}")
        print(f"{'='*50}")

        train = pd.read_csv(DATA_DIR / f"stock_{i}_train.csv")
        test  = pd.read_csv(DATA_DIR / f"stock_{i}_test.csv")
        X = train.drop("target", axis=1)
        y = train["target"]
        profile = profile_dataset(train, test)
        result  = select_model(profile, X, y)
        pred    = result["model"].predict(test)[0]

        print(f"  Prediction:   {pred:.2f}")
        print(f"  Noise ratio:  {result['noise_ratio']:.4f}")
        print(f"  Best params:  {result['best_params']}")
