import warnings
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor, VotingRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

SCORING = "neg_root_mean_squared_error"
DEFAULT_RANDOM_STATE = 42
LINEAR_LIKE_FAMILIES = {"linear", "robust_linear", "latent"}

MODEL_COMPLEXITY_RANK = {
    "LinearRegression": 0,
    "BayesianRidge": 1,
    "Ridge": 2,
    "HuberRegressor": 3,
    "ElasticNet": 4,
    "PLSRegression": 5,
    "Ridge+Poly": 6,
    "KNN": 7,
    "SVR-Linear": 8,
    "SVR-RBF": 9,
    "GradientBoosting": 10,
    "LightGBM": 11,
    "XGBoost": 12,
}


@dataclass(frozen=True)
class CVPlan:
    kind: str
    n_splits: int
    n_repeats: int = 1

    def build(self, random_state: int):
        if self.kind == "repeated":
            return RepeatedKFold(
                n_splits=self.n_splits,
                n_repeats=self.n_repeats,
                random_state=random_state,
            )
        return KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=random_state,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "n_splits": self.n_splits,
            "n_repeats": self.n_repeats,
        }


@dataclass(frozen=True)
class MixPlan:
    top_k: int = 3
    qualified_rel_gap: float = 0.08
    max_qualified_models: int = 5
    allow_stacking: bool = False
    allow_linear_only: bool = False
    allow_linear_only_all: bool = False
    allow_all_candidates_vote: bool = True
    all_candidates_max: int = 6
    max_stack_models: int = 4

    def as_dict(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "qualified_rel_gap": self.qualified_rel_gap,
            "max_qualified_models": self.max_qualified_models,
            "allow_stacking": self.allow_stacking,
            "allow_linear_only": self.allow_linear_only,
            "allow_linear_only_all": self.allow_linear_only_all,
            "allow_all_candidates_vote": self.allow_all_candidates_vote,
            "all_candidates_max": self.all_candidates_max,
            "max_stack_models": self.max_stack_models,
        }


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    estimator: Any
    param_grid: dict[str, Any]


@dataclass(frozen=True)
class StockStrategy:
    stock_number: int | None
    tuning_cv: CVPlan
    comparison_cv: CVPlan
    mix_plan: MixPlan
    candidates: list[CandidateSpec]


@dataclass
class CandidateResult:
    name: str
    family: str
    best_estimator: Any
    best_params: dict[str, Any]
    tune_rmse: float
    tune_std: float
    train_rmse: float = 0.0
    cv_rmse: float = 0.0
    cv_std: float = 0.0
    overfit_ratio: float = 1.0
    overfit_flag: bool = False

    def summary(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "best_params": self.best_params,
            "tune_rmse": self.tune_rmse,
            "tune_std": self.tune_std,
            "train_rmse": self.train_rmse,
            "final_rmse": self.cv_rmse,
            "final_std": self.cv_std,
            "overfit_ratio": self.overfit_ratio,
            "overfit_flag": self.overfit_flag,
        }


@dataclass
class EnsembleResult:
    name: str
    method: str
    members: list[str]
    estimator: Any
    cv_rmse: float
    cv_std: float
    weights: list[float] | None = None
    train_rmse: float = 0.0
    overfit_ratio: float = 1.0
    overfit_flag: bool = False


def _resolve_stock_number(stock_number):
    if stock_number is None:
        return None
    try:
        stock_number = int(stock_number)
    except (TypeError, ValueError):
        return None
    return stock_number if 1 <= stock_number <= 9 else None


def _scaled_estimator(step_name: str, estimator: Any):
    return Pipeline([
        ("scaler", StandardScaler()),
        (step_name, estimator),
    ])


def _candidate(name: str, family: str, estimator: Any, param_grid: dict[str, Any]) -> CandidateSpec:
    return CandidateSpec(name=name, family=family, estimator=estimator, param_grid=param_grid)


def _linear_candidate():
    return _candidate("LinearRegression", "linear", _scaled_estimator("linear", LinearRegression()), {})


def _ridge_candidate(alphas):
    return _candidate(
        "Ridge",
        "linear",
        _scaled_estimator("ridge", Ridge()),
        {"ridge__alpha": alphas},
    )


def _bayesian_ridge_candidate(alpha_values, lambda_values):
    return _candidate(
        "BayesianRidge",
        "linear",
        _scaled_estimator("bayes", BayesianRidge()),
        {
            "bayes__alpha_1": alpha_values,
            "bayes__lambda_1": lambda_values,
        },
    )


def _elastic_net_candidate(alphas, l1_ratios, random_state):
    return _candidate(
        "ElasticNet",
        "linear",
        _scaled_estimator("enet", ElasticNet(max_iter=10000, random_state=random_state)),
        {
            "enet__alpha": alphas,
            "enet__l1_ratio": l1_ratios,
        },
    )


def _huber_candidate(alphas, epsilons):
    return _candidate(
        "HuberRegressor",
        "robust_linear",
        _scaled_estimator("huber", HuberRegressor(max_iter=2000)),
        {
            "huber__alpha": alphas,
            "huber__epsilon": epsilons,
        },
    )


def _pls_candidate(components):
    return _candidate(
        "PLSRegression",
        "latent",
        PLSRegression(scale=True),
        {"n_components": components},
    )


def _ridge_poly_candidate(degrees, alphas):
    return _candidate(
        "Ridge+Poly",
        "poly",
        Pipeline([
            ("poly", PolynomialFeatures(include_bias=False)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge()),
        ]),
        {
            "poly__degree": degrees,
            "ridge__alpha": alphas,
        },
    )


def _svr_linear_candidate(c_values, epsilons):
    return _candidate(
        "SVR-Linear",
        "svm",
        _scaled_estimator("svr", SVR(kernel="linear")),
        {
            "svr__C": c_values,
            "svr__epsilon": epsilons,
        },
    )


def _svr_rbf_candidate(c_values, epsilons, gammas):
    return _candidate(
        "SVR-RBF",
        "svm",
        _scaled_estimator("svr", SVR(kernel="rbf")),
        {
            "svr__C": c_values,
            "svr__epsilon": epsilons,
            "svr__gamma": gammas,
        },
    )


def _knn_candidate(neighbors, weights):
    return _candidate(
        "KNN",
        "local",
        _scaled_estimator("knn", KNeighborsRegressor()),
        {
            "knn__n_neighbors": neighbors,
            "knn__weights": weights,
        },
    )


def _gradient_boosting_candidate(random_state, losses, n_estimators, learning_rates, depths, min_samples_leaf):
    return _candidate(
        "GradientBoosting",
        "tree",
        GradientBoostingRegressor(random_state=random_state),
        {
            "loss": losses,
            "n_estimators": n_estimators,
            "learning_rate": learning_rates,
            "max_depth": depths,
            "min_samples_leaf": min_samples_leaf,
        },
    )


def _lightgbm_candidate(random_state, n_estimators, num_leaves, learning_rates, min_child_samples, reg_lambda):
    return _candidate(
        "LightGBM",
        "tree",
        lgb.LGBMRegressor(
            objective="regression",
            random_state=random_state,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1,
            n_jobs=1,
        ),
        {
            "n_estimators": n_estimators,
            "num_leaves": num_leaves,
            "learning_rate": learning_rates,
            "min_child_samples": min_child_samples,
            "reg_lambda": reg_lambda,
        },
    )


def _xgboost_candidate(random_state, n_estimators, depths, learning_rates, min_child_weight, reg_lambda):
    return _candidate(
        "XGBoost",
        "tree",
        XGBRegressor(
            objective="reg:squarederror",
            random_state=random_state,
            verbosity=0,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=1,
        ),
        {
            "n_estimators": n_estimators,
            "max_depth": depths,
            "learning_rate": learning_rates,
            "min_child_weight": min_child_weight,
            "reg_lambda": reg_lambda,
        },
    )


def _default_cv_plan(profile) -> CVPlan:
    if profile.n_train < 40:
        return CVPlan(kind="repeated", n_splits=3, n_repeats=12)
    if profile.n_train < 120:
        return CVPlan(kind="repeated", n_splits=4, n_repeats=8)
    return CVPlan(kind="kfold", n_splits=5)


def _mix_plan_with(**overrides) -> MixPlan:
    base = MixPlan().as_dict()
    base.update(overrides)
    return MixPlan(**base)


def _stock_strategy(stock_number, candidates, cv_plan, mix_plan=None, comparison_cv=None) -> StockStrategy:
    return StockStrategy(
        stock_number=stock_number,
        tuning_cv=cv_plan,
        comparison_cv=comparison_cv or cv_plan,
        mix_plan=mix_plan or MixPlan(),
        candidates=candidates,
    )


def _stock_1_strategy(random_state) -> StockStrategy:
    return _stock_strategy(
        1,
        [
            _linear_candidate(),
            _ridge_candidate([1e-4, 1e-3, 1e-2, 1e-1, 1, 10]),
            _bayesian_ridge_candidate([1e-7, 1e-6, 1e-5], [1e-7, 1e-6, 1e-5]),
            _ridge_poly_candidate([2, 3], [0.01, 0.1, 1, 10]),
            _gradient_boosting_candidate(random_state, ["squared_error"], [100, 200, 400], [0.03, 0.05, 0.1], [2, 3], [3, 5]),
            _xgboost_candidate(random_state, [150, 300], [2, 3, 4], [0.03, 0.05, 0.1], [1, 3], [1, 3, 10]),
        ],
        cv_plan=CVPlan(kind="kfold", n_splits=5),
        mix_plan=_mix_plan_with(top_k=4, qualified_rel_gap=0.10, max_qualified_models=6, allow_stacking=True),
    )


def _stock_2_strategy(random_state) -> StockStrategy:
    return _stock_strategy(
        2,
        [
            _ridge_candidate([0.1, 1, 10, 100, 1000]),
            _bayesian_ridge_candidate([1e-6, 1e-5, 1e-4], [1e-6, 1e-5, 1e-4]),
            _elastic_net_candidate([5e-4, 1e-3, 1e-2, 1e-1], [0.1, 0.3, 0.5, 0.8], random_state),
            _huber_candidate([1e-4, 1e-3, 1e-2], [1.35, 1.75]),
            _lightgbm_candidate(random_state, [100, 200, 400], [15, 31, 63], [0.03, 0.05, 0.1], [10, 20, 40], [1, 3, 10]),
            _xgboost_candidate(random_state, [100, 200, 400], [2, 3, 4], [0.03, 0.05, 0.1], [1, 3, 5], [1, 3, 10]),
        ],
        cv_plan=CVPlan(kind="kfold", n_splits=5),
        mix_plan=_mix_plan_with(top_k=4, qualified_rel_gap=0.08, allow_stacking=True),
    )


def _stock_3_strategy(random_state) -> StockStrategy:
    return _stock_strategy(
        3,
        [
            _linear_candidate(),
            _ridge_candidate([0.3, 1, 3, 10, 30, 100]),
            _bayesian_ridge_candidate([1e-6, 1e-5], [1e-6, 1e-5]),
            _elastic_net_candidate([0.03, 0.05, 0.1, 0.2, 0.5], [0.05, 0.1, 0.2, 0.3, 0.5], random_state),
            _huber_candidate([1e-4, 1e-3, 1e-2], [1.1, 1.35, 1.75]),
            _svr_linear_candidate([0.05, 0.1, 0.2, 0.5, 1, 2], [0.1, 0.3, 0.5, 1.0]),
        ],
        cv_plan=CVPlan(kind="repeated", n_splits=5, n_repeats=20),
        mix_plan=_mix_plan_with(
            top_k=1,
            qualified_rel_gap=0.0,
            max_qualified_models=1,
            allow_all_candidates_vote=False,
        ),
    )


def _stock_4_strategy(random_state) -> StockStrategy:
    return _stock_strategy(
        4,
        [
            _gradient_boosting_candidate(random_state, ["huber"], [100, 200, 400], [0.03, 0.05, 0.1], [2, 3], [3, 5, 10]),
            _huber_candidate([1e-4, 1e-3, 1e-2], [1.35, 1.75]),
            _ridge_candidate([0.1, 1, 10, 100, 1000]),
            _elastic_net_candidate([0.001, 0.01, 0.1, 1], [0.1, 0.3, 0.5, 0.8], random_state),
            _bayesian_ridge_candidate([1e-6, 1e-5, 1e-4], [1e-6, 1e-5, 1e-4]),
            _lightgbm_candidate(random_state, [100, 200, 400], [15, 31, 63], [0.03, 0.05, 0.1], [10, 20, 40], [1, 3, 10]),
            _xgboost_candidate(random_state, [100, 200, 400], [2, 3, 4], [0.03, 0.05, 0.1], [1, 3, 5], [1, 3, 10]),
        ],
        cv_plan=CVPlan(kind="kfold", n_splits=5),
        mix_plan=_mix_plan_with(top_k=4, qualified_rel_gap=0.10, max_qualified_models=6, allow_stacking=True, all_candidates_max=7),
    )


def _stock_5_strategy(random_state) -> StockStrategy:
    return _stock_strategy(
        5,
        [
            _huber_candidate([1e-4, 1e-3, 1e-2], [1.1, 1.35, 1.75]),
            _bayesian_ridge_candidate([1e-6, 1e-5, 1e-4], [1e-6, 1e-5, 1e-4]),
            _ridge_candidate([0.1, 1, 10, 100, 1000]),
            _elastic_net_candidate([0.001, 0.01, 0.1, 1], [0.1, 0.3, 0.5, 0.8], random_state),
            _gradient_boosting_candidate(random_state, ["huber"], [100, 200, 400], [0.03, 0.05, 0.1], [2, 3], [3, 5, 10]),
            _lightgbm_candidate(random_state, [100, 200, 300], [15, 31], [0.03, 0.05, 0.1], [10, 20, 40], [1, 3, 10]),
        ],
        cv_plan=CVPlan(kind="kfold", n_splits=5),
        mix_plan=_mix_plan_with(top_k=3, qualified_rel_gap=0.08, allow_stacking=True),
    )


def _stock_6_strategy(random_state) -> StockStrategy:
    return _stock_strategy(
        6,
        [
            _svr_rbf_candidate([0.1, 0.5, 1, 5, 10, 25], [0.01, 0.1, 0.5, 1.0], ["scale", 0.05, 0.1, 0.2, 0.5]),
            _svr_linear_candidate([0.1, 0.5, 1, 5, 10, 25], [0.01, 0.1, 0.5, 1.0]),
            _huber_candidate([1e-4, 1e-3], [1.35, 1.75]),
            _bayesian_ridge_candidate([1e-6, 1e-5, 1e-4], [1e-6, 1e-5, 1e-4]),
            _ridge_candidate([1, 10, 100, 1000]),
        ],
        cv_plan=CVPlan(kind="repeated", n_splits=4, n_repeats=10),
        mix_plan=_mix_plan_with(top_k=2, qualified_rel_gap=0.06, max_qualified_models=3),
    )


def _stock_7_strategy(random_state) -> StockStrategy:
    return _stock_strategy(
        7,
        [
            _huber_candidate([1e-4, 1e-3, 1e-2], [1.35, 1.75]),
            _elastic_net_candidate([0.01, 0.1, 1, 10], [0.1, 0.3, 0.5, 0.8], random_state),
            _bayesian_ridge_candidate([1e-6, 1e-5, 1e-4], [1e-6, 1e-5, 1e-4]),
            _ridge_candidate([1, 10, 100, 1000, 10000]),
            _gradient_boosting_candidate(random_state, ["huber"], [100, 200], [0.03, 0.05], [1, 2], [5, 10]),
        ],
        cv_plan=CVPlan(kind="kfold", n_splits=5),
        mix_plan=_mix_plan_with(top_k=2, qualified_rel_gap=0.04, max_qualified_models=4, allow_linear_only=True, allow_linear_only_all=True, all_candidates_max=5),
    )


def _stock_8_strategy(random_state) -> StockStrategy:
    return _stock_strategy(
        8,
        [
            _bayesian_ridge_candidate([1e-6, 1e-5, 1e-4], [1e-6, 1e-5, 1e-4]),
            _pls_candidate([2, 3, 4, 5]),
            _elastic_net_candidate([0.001, 0.01, 0.1, 1], [0.1, 0.3, 0.5, 0.8], random_state),
            _ridge_candidate([1, 10, 100, 1000]),
            _huber_candidate([1e-4, 1e-3], [1.35, 1.75]),
        ],
        cv_plan=CVPlan(kind="kfold", n_splits=5),
        mix_plan=_mix_plan_with(top_k=3, qualified_rel_gap=0.05, max_qualified_models=4, allow_linear_only=True, allow_linear_only_all=True, all_candidates_max=5),
    )


def _stock_9_strategy(random_state) -> StockStrategy:
    return _stock_strategy(
        9,
        [
            _svr_rbf_candidate([0.1, 0.5, 1, 5, 10], [0.01, 0.1, 0.5, 1.0], ["scale", 0.05, 0.1, 0.2]),
            _bayesian_ridge_candidate([1e-6, 1e-5, 1e-4], [1e-6, 1e-5, 1e-4]),
            _svr_linear_candidate([0.1, 0.5, 1, 5, 10], [0.01, 0.1, 0.5, 1.0]),
            _pls_candidate([2, 3]),
            _elastic_net_candidate([0.01, 0.1, 1], [0.2, 0.5, 0.8], random_state),
            _ridge_candidate([1, 10, 100, 1000]),
        ],
        cv_plan=CVPlan(kind="repeated", n_splits=3, n_repeats=20),
        mix_plan=_mix_plan_with(top_k=2, qualified_rel_gap=0.06, max_qualified_models=3),
    )


def _generic_candidates(profile, random_state):
    candidates = [
        _ridge_candidate([0.001, 0.01, 0.1, 1, 10, 100]),
        _bayesian_ridge_candidate([1e-6, 1e-5, 1e-4], [1e-6, 1e-5, 1e-4]),
        _elastic_net_candidate([0.001, 0.01, 0.1, 1], [0.2, 0.5, 0.8], random_state),
        _huber_candidate([1e-4, 1e-3], [1.35, 1.75]),
    ]
    if profile.try_poly_features:
        candidates.append(_ridge_poly_candidate([2], [0.1, 1, 10]))
    if profile.high_collinearity and profile.n_features >= 10:
        max_components = min(5, profile.n_features)
        candidates.append(_pls_candidate(list(range(2, max_components + 1))))
    if profile.n_train <= 500:
        candidates.append(_svr_linear_candidate([0.1, 1, 10], [0.01, 0.1, 0.5]))
        candidates.append(_svr_rbf_candidate([0.1, 1, 10], [0.01, 0.1, 0.5], ["scale", 0.1, 0.5]))
    if profile.n_train >= 80:
        losses = ["huber"] if profile.use_robust_loss else ["squared_error"]
        candidates.append(_gradient_boosting_candidate(random_state, losses, [100, 200], [0.05, 0.1], [2, 3], [3, 5]))
    return candidates


def _build_strategy(profile, stock_number, random_state) -> StockStrategy:
    stock_number = _resolve_stock_number(stock_number)
    builders = {
        1: _stock_1_strategy,
        2: _stock_2_strategy,
        3: _stock_3_strategy,
        4: _stock_4_strategy,
        5: _stock_5_strategy,
        6: _stock_6_strategy,
        7: _stock_7_strategy,
        8: _stock_8_strategy,
        9: _stock_9_strategy,
    }
    if stock_number in builders:
        return builders[stock_number](random_state)
    cv_plan = _default_cv_plan(profile)
    return _stock_strategy(
        stock_number,
        _generic_candidates(profile, random_state),
        cv_plan=cv_plan,
        mix_plan=MixPlan(),
    )


def _model_complexity_rank(model_name):
    return MODEL_COMPLEXITY_RANK.get(model_name, 99)


def _relative_gap(score, best_score):
    if best_score <= 0:
        return 0.0 if score <= 0 else float("inf")
    return (score - best_score) / best_score


def _inverse_rmse_weights(candidates):
    rmse_values = np.array([candidate.cv_rmse for candidate in candidates], dtype=float)
    safe_rmse = np.maximum(rmse_values, 1e-9)
    return (1.0 / safe_rmse).tolist()


def _is_linear_like(candidate):
    return candidate.family in LINEAR_LIKE_FAMILIES


def _has_family_diversity(candidates, allow_linear_only=False):
    families = {candidate.family for candidate in candidates}
    if len(families) < 2:
        return False
    if allow_linear_only:
        return True
    return not all(_is_linear_like(candidate) for candidate in candidates)


def _select_diverse_members(ordered_candidates, max_models, allow_linear_only):
    selected = []
    seen_families = set()
    for candidate in ordered_candidates:
        if candidate.family in seen_families:
            continue
        selected.append(candidate)
        seen_families.add(candidate.family)
        if len(selected) >= max_models:
            break
    if len(selected) < 2 or not _has_family_diversity(selected, allow_linear_only):
        return []
    return selected


def _select_best_with_tiebreak(comparison):
    best_rmse = min(scores["cv_rmse"] for scores in comparison.values())
    best_std = min(scores["cv_std"] for scores in comparison.values())
    tie_tolerance = max(0.05, best_rmse * 0.003, min(best_std, 0.25))
    contenders = {
        name: scores
        for name, scores in comparison.items()
        if scores["cv_rmse"] <= best_rmse + tie_tolerance
    }
    best_name = min(
        contenders,
        key=lambda name: (
            _model_complexity_rank(name),
            contenders[name]["cv_rmse"],
            contenders[name]["cv_std"],
        ),
    )
    return best_name, tie_tolerance


class StockModelSelector:
    def __init__(self, profile, X_train, y_train, stock_number=None, verbose=True, random_state=DEFAULT_RANDOM_STATE):
        self.profile = profile
        self.X_train = X_train
        self.y_train = y_train
        self.stock_number = _resolve_stock_number(stock_number)
        self.verbose = verbose
        self.random_state = random_state
        self.strategy = _build_strategy(profile, self.stock_number, random_state)

    def select(self) -> dict[str, Any]:
        tuned_results = self._tune_candidates()
        best_single_name, best_single_scores = self._compare_candidates(tuned_results)
        ordered_candidates = self._ordered_candidates(tuned_results)
        best_ensemble, ensemble_results = self._evaluate_ensembles(ordered_candidates)
        return self._build_selection_result(
            tuned_results=tuned_results,
            ordered_candidates=ordered_candidates,
            best_single_name=best_single_name,
            best_single_scores=best_single_scores,
            best_ensemble=best_ensemble,
            ensemble_results=ensemble_results,
        )

    def _log(self, message: str):
        if self.verbose:
            print(message)

    def _compute_train_metrics(self, estimator, cv_rmse: float, fit_before_predict: bool = False):
        if fit_before_predict:
            estimator.fit(self.X_train, self.y_train)

        predictions = np.asarray(estimator.predict(self.X_train), dtype=float).reshape(-1)
        y_true = np.asarray(self.y_train, dtype=float).reshape(-1)
        train_rmse = float(np.sqrt(np.mean((y_true - predictions) ** 2)))
        overfit_ratio = train_rmse / cv_rmse if cv_rmse > 0 else 1.0
        overfit_flag = overfit_ratio < 0.85
        return train_rmse, overfit_ratio, overfit_flag

    def _tune_candidates(self) -> dict[str, CandidateResult]:
        tuning_cv = self.strategy.tuning_cv.build(self.random_state)
        self._log(f"\n  {'Model':<20} {'Tune RMSE':>10}  {'+/- std':>8}  {'Train':>8}  {'O/F':>6}")
        self._log(f"  {'-' * 72}")

        tuned_results = {}
        for spec in self.strategy.candidates:
            result = self._tune_candidate(spec, tuning_cv)
            tuned_results[result.name] = result
            tune_overfit_ratio = result.train_rmse / result.tune_rmse if result.tune_rmse > 0 else 1.0
            self._log(
                f"  {result.name:<20} {result.tune_rmse:>10.3f}  {result.tune_std:>8.3f}  "
                f"{result.train_rmse:>8.3f}  {tune_overfit_ratio:>6.3f}"
            )
        return tuned_results

    def _tune_candidate(self, spec: CandidateSpec, cv) -> CandidateResult:
        search = GridSearchCV(
            estimator=spec.estimator,
            param_grid=spec.param_grid,
            scoring=SCORING,
            cv=cv,
            refit=True,
            n_jobs=1,
        )
        search.fit(self.X_train, self.y_train)

        best_index = search.best_index_
        best_rmse = float(-search.best_score_)
        best_std = float(search.cv_results_["std_test_score"][best_index])
        train_rmse, _, _ = self._compute_train_metrics(search.best_estimator_, best_rmse)

        return CandidateResult(
            name=spec.name,
            family=spec.family,
            best_estimator=search.best_estimator_,
            best_params=search.best_params_,
            tune_rmse=best_rmse,
            tune_std=best_std,
            train_rmse=train_rmse,
        )

    def _compare_candidates(self, tuned_results: dict[str, CandidateResult]):
        comparison_cv = self.strategy.comparison_cv.build(self.random_state + 1)
        comparison = {}

        self._log(f"\n  {'Model':<20} {'Final RMSE':>10}  {'+/- std':>8}  {'Train':>8}  {'O/F':>6}")
        self._log(f"  {'-' * 73}")

        for result in tuned_results.values():
            scores = cross_val_score(
                result.best_estimator,
                self.X_train,
                self.y_train,
                scoring=SCORING,
                cv=comparison_cv,
            )
            result.cv_rmse = float(-scores.mean())
            result.cv_std = float(scores.std())
            result.train_rmse, result.overfit_ratio, result.overfit_flag = self._compute_train_metrics(
                result.best_estimator,
                result.cv_rmse,
            )
            comparison[result.name] = {
                "cv_rmse": result.cv_rmse,
                "cv_std": result.cv_std,
                "train_rmse": result.train_rmse,
                "overfit_ratio": result.overfit_ratio,
                "overfit_flag": result.overfit_flag,
            }
            self._log(
                f"  {result.name:<20} {result.cv_rmse:>10.3f}  {result.cv_std:>8.3f}  "
                f"{result.train_rmse:>8.3f}  {result.overfit_ratio:>6.3f}"
            )

        best_name, tie_tolerance = _select_best_with_tiebreak(comparison)
        comparison[best_name]["tie_tolerance"] = tie_tolerance
        return best_name, comparison[best_name]

    def _ordered_candidates(self, tuned_results: dict[str, CandidateResult]) -> list[CandidateResult]:
        return sorted(
            tuned_results.values(),
            key=lambda result: (
                result.cv_rmse,
                result.cv_std,
                _model_complexity_rank(result.name),
            ),
        )

    def _evaluate_ensembles(self, ordered_candidates: list[CandidateResult]):
        ensemble_results = {}
        evaluated_specs = set()
        if len(ordered_candidates) < 2:
            return None, ensemble_results

        mix_plan = self.strategy.mix_plan
        best_single_rmse = ordered_candidates[0].cv_rmse
        stacking_cv = KFold(
            n_splits=self.strategy.comparison_cv.n_splits,
            shuffle=True,
            random_state=self.random_state + 2,
        )

        self._log(f"\n  {'Ensemble':<36} {'RMSE':>10}  {'+/- std':>8}  {'Train':>8}  {'O/F':>6}")
        self._log(f"  {'-' * 85}")

        candidate_sets = []
        max_top = min(mix_plan.top_k, len(ordered_candidates))
        for n_models in range(2, max_top + 1):
            members = ordered_candidates[:n_models]
            if _has_family_diversity(members, mix_plan.allow_linear_only):
                candidate_sets.append((f"top{n_models}", members))

        diverse_members = _select_diverse_members(
            ordered_candidates,
            max_models=min(mix_plan.top_k, len(ordered_candidates)),
            allow_linear_only=mix_plan.allow_linear_only,
        )
        if diverse_members:
            candidate_sets.append(("diverse", diverse_members))

        qualified_members = [
            candidate
            for candidate in ordered_candidates
            if _relative_gap(candidate.cv_rmse, best_single_rmse) <= mix_plan.qualified_rel_gap
        ][:mix_plan.max_qualified_models]
        if len(qualified_members) >= 2 and _has_family_diversity(qualified_members, mix_plan.allow_linear_only):
            candidate_sets.append(("qualified", qualified_members))

        if mix_plan.allow_all_candidates_vote and len(ordered_candidates) >= 2:
            all_members = ordered_candidates[:mix_plan.all_candidates_max]
            if _has_family_diversity(all_members, mix_plan.allow_linear_only_all):
                candidate_sets.append(("all", all_members))

        comparison_cv = self.strategy.comparison_cv.build(self.random_state + 1)
        for label, members in candidate_sets:
            for ensemble in self._ensemble_specs(label, members, stacking_cv):
                spec_key = (ensemble.method, tuple(ensemble.members))
                if spec_key in evaluated_specs:
                    continue
                evaluated_specs.add(spec_key)

                scores = cross_val_score(
                    ensemble.estimator,
                    self.X_train,
                    self.y_train,
                    scoring=SCORING,
                    cv=comparison_cv,
                )
                ensemble.cv_rmse = float(-scores.mean())
                ensemble.cv_std = float(scores.std())
                ensemble.train_rmse, ensemble.overfit_ratio, ensemble.overfit_flag = self._compute_train_metrics(
                    ensemble.estimator,
                    ensemble.cv_rmse,
                    fit_before_predict=True,
                )
                ensemble_results[ensemble.name] = ensemble
                self._log(
                    f"  {ensemble.name:<36} {ensemble.cv_rmse:>10.3f}  {ensemble.cv_std:>8.3f}  "
                    f"{ensemble.train_rmse:>8.3f}  {ensemble.overfit_ratio:>6.3f}"
                )

        if not ensemble_results:
            return None, ensemble_results

        best_name = min(
            ensemble_results,
            key=lambda name: (
                ensemble_results[name].cv_rmse,
                ensemble_results[name].cv_std,
            ),
        )
        return ensemble_results[best_name], ensemble_results

    def _ensemble_specs(self, label: str, members: list[CandidateResult], stacking_cv) -> list[EnsembleResult]:
        estimators = [(candidate.name, clone(candidate.best_estimator)) for candidate in members]
        weights = _inverse_rmse_weights(members)
        member_names = [candidate.name for candidate in members]
        specs = [
            EnsembleResult(
                name=f"SimpleAverage({label})",
                method="simple_average",
                members=member_names,
                estimator=VotingRegressor(estimators=estimators, weights=[1.0] * len(members)),
                weights=[1.0] * len(members),
                cv_rmse=0.0,
                cv_std=0.0,
            ),
            EnsembleResult(
                name=f"WeightedAverage({label})",
                method="weighted_average",
                members=member_names,
                estimator=VotingRegressor(estimators=estimators, weights=weights),
                weights=weights,
                cv_rmse=0.0,
                cv_std=0.0,
            ),
        ]

        if self.strategy.mix_plan.allow_stacking and len(members) <= self.strategy.mix_plan.max_stack_models:
            specs.append(
                EnsembleResult(
                    name=f"Stacking({label})",
                    method="stacking",
                    members=member_names,
                    estimator=StackingRegressor(
                        estimators=estimators,
                        final_estimator=Ridge(alpha=1.0),
                        cv=stacking_cv,
                        n_jobs=1,
                    ),
                    weights=None,
                    cv_rmse=0.0,
                    cv_std=0.0,
                )
            )
        return specs

    def _build_selection_result(
        self,
        tuned_results: dict[str, CandidateResult],
        ordered_candidates: list[CandidateResult],
        best_single_name: str,
        best_single_scores: dict[str, float],
        best_ensemble: EnsembleResult | None,
        ensemble_results: dict[str, EnsembleResult],
    ) -> dict[str, Any]:
        best_single_result = tuned_results[best_single_name]
        final_name = best_single_name
        final_model = clone(best_single_result.best_estimator)
        final_best_params = best_single_result.best_params
        final_rmse = best_single_scores["cv_rmse"]
        final_std = best_single_scores["cv_std"]
        mixing_used = False
        mixing_method = None
        mixing_members = []
        mixing_weights = None

        if best_ensemble and best_ensemble.cv_rmse < final_rmse:
            final_name = best_ensemble.name
            final_model = clone(best_ensemble.estimator)
            final_best_params = {
                "ensemble_method": best_ensemble.method,
                "members": best_ensemble.members,
            }
            if best_ensemble.weights is not None:
                final_best_params["weights"] = best_ensemble.weights
            final_rmse = best_ensemble.cv_rmse
            final_std = best_ensemble.cv_std
            mixing_used = True
            mixing_method = best_ensemble.method
            mixing_members = best_ensemble.members
            mixing_weights = best_ensemble.weights

        final_model.fit(self.X_train, self.y_train)
        train_predictions = final_model.predict(self.X_train)
        train_rmse = float(np.sqrt(np.mean((np.asarray(self.y_train) - train_predictions) ** 2)))
        overfit_ratio = train_rmse / final_rmse if final_rmse > 0 else 1.0
        overfit_flag = overfit_ratio < 0.85

        noise_ratio = final_rmse / self.profile.target_std if self.profile.target_std > 0 else 1.0
        self.profile.noise_ratio = noise_ratio
        self.profile.no_quote = noise_ratio > 0.98

        if self.verbose:
            if mixing_used:
                print(f"\n  Winner: {final_name}  (RMSE={final_rmse:.3f} +/- {final_std:.3f})")
                print(f"  Ensemble beat best single model {best_single_name} ({best_single_scores['cv_rmse']:.3f})")
            else:
                print(f"\n  Winner: {best_single_name}  (RMSE={final_rmse:.3f} +/- {final_std:.3f})")
            print(f"  Train RMSE: {train_rmse:.3f}  |  Overfit ratio: {overfit_ratio:.3f}")

        return {
            "name": final_name,
            "model": final_model,
            "best_params": final_best_params,
            "cv_rmse": final_rmse,
            "cv_std": final_std,
            "all_scores": {candidate.name: candidate.cv_rmse for candidate in ordered_candidates},
            "mixing_scores": {name: result.cv_rmse for name, result in ensemble_results.items()},
            "tuned_models": {name: result.summary() for name, result in tuned_results.items()},
            "ensemble_models": {
                name: {
                    "method": result.method,
                    "members": result.members,
                    "weights": result.weights,
                    "final_rmse": result.cv_rmse,
                    "final_std": result.cv_std,
                    "train_rmse": result.train_rmse,
                    "overfit_ratio": result.overfit_ratio,
                    "overfit_flag": result.overfit_flag,
                }
                for name, result in ensemble_results.items()
            },
            "noise_ratio": noise_ratio,
            "blended": mixing_used,
            "mixing_used": mixing_used,
            "mixing_method": mixing_method,
            "mixing_members": mixing_members,
            "mixing_weights": mixing_weights,
            "best_single_model": best_single_name,
            "best_single_rmse": best_single_scores["cv_rmse"],
            "train_rmse": train_rmse,
            "overfit_ratio": overfit_ratio,
            "overfit_flag": overfit_flag,
            "tie_tolerance": best_single_scores.get("tie_tolerance", 0.0),
            "stock_number": self.stock_number,
            "candidate_models": [candidate.name for candidate in ordered_candidates],
            "mix_plan": self.strategy.mix_plan.as_dict(),
            "tuning_cv_plan": self.strategy.tuning_cv.as_dict(),
            "comparison_cv_plan": self.strategy.comparison_cv.as_dict(),
            "random_state": self.random_state,
        }


def select_model(profile, X_train, y_train, stock_number=None, verbose=True, random_state=DEFAULT_RANDOM_STATE):
    selector = StockModelSelector(
        profile=profile,
        X_train=X_train,
        y_train=y_train,
        stock_number=stock_number,
        verbose=verbose,
        random_state=random_state,
    )
    return selector.select()


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
        result = select_model(profile, X, y, stock_number=i)
        pred = result["model"].predict(test)[0]
        print(f"  Prediction:   {pred:.2f}")
        print(f"  Noise ratio:  {result['noise_ratio']:.4f}")
        print(f"  Best params:  {result['best_params']}")
