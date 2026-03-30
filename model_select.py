import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV, ElasticNetCV, BayesianRidge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
import lightgbm as lgb
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

#Build a set of candidate models based on the dataset profile
def _build_candidates(profile):

    candidates = {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000]))
        ]),

        "BayesianRidge": Pipeline([
            ("scaler", StandardScaler()),
            ("br", BayesianRidge())
        ]),

        "ElasticNet": Pipeline([
            ("scaler", StandardScaler()),
            ("en", ElasticNetCV(
                l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                alphas=[0.001, 0.01, 0.1, 1, 10],
                cv=3, random_state=42, max_iter=5000
            ))
        ]),
    }

    # Only include poly if profile says it's worth trying
    if profile.try_poly_features:
        candidates["Ridge+Poly"] = Pipeline([
            ("poly",   PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("ridge",  RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000]))
        ])

    # Only include tree models on larger datasets where they can generalise
    if profile.n_train >= 80:

        candidates["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=200,       
            num_leaves=15,         
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,          
            reg_lambda=2.0,         
            min_child_samples=max(5, profile.n_train // 20),
            random_state=42,
            verbose=-1
        )

        candidates["XGBoost"] = XGBRegressor(
            n_estimators=200,       
            max_depth=3,            
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=42,
            verbosity=0
        )

    return candidates

def select_model(profile, X_train, y_train, verbose=True):

    candidates = _build_candidates(profile)

    # CV setup
    n_folds = min(5, max(3, profile.n_train // 10))
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Run CV — collect per-fold predictions for potential blending
    scores = {}
    fold_preds = {}

    if verbose:
        print(f"\n  {'Model':<20} {'RMSE':>8}  {'±std':>6}")
        print(f"  {'-'*38}")

    for name, model in candidates.items():
        cv_scores = cross_val_score(
            model, X_train, y_train,
            scoring="neg_root_mean_squared_error",
            cv=cv
        )
        rmse = float(-cv_scores.mean())
        std  = float(cv_scores.std())
        scores[name] = (rmse, std)

        if verbose:
            print(f"  {name:<20} {rmse:>8.3f}  {std:>6.3f}")

    # Pick winner
    best_name = min(scores, key=lambda k: scores[k][0])
    best_rmse, best_std = scores[best_name]

    # --- NEW: simple blend of top-2 if they're close ---
    sorted_models = sorted(scores.items(), key=lambda kv: kv[1][0])
    first_name,  (first_rmse,  _) = sorted_models[0]
    second_name, (second_rmse, _) = sorted_models[1]
    gap_pct = (second_rmse - first_rmse) / first_rmse * 100

    # Only blend when: (1) gap is genuinely tight, (2) the models aren't both terrible
    # noise_ratio computed here early to gate blending
    noise_ratio = first_rmse / profile.target_std if profile.target_std > 0 else 1.0
    use_blend = gap_pct < 1.5 and noise_ratio < 0.7

    if use_blend:
        m1 = candidates[first_name]
        m2 = candidates[second_name]
        m1.fit(X_train, y_train)
        m2.fit(X_train, y_train)

        # Weight inversely by RMSE
        w1 = 1.0 / first_rmse
        w2 = 1.0 / second_rmse
        w_total = w1 + w2
        w1, w2 = w1 / w_total, w2 / w_total

        blend_name = f"Blend({first_name}+{second_name})"
        if verbose:
            print(f"\n  Top-2 within {gap_pct:.1f}% — blending {first_name} ({w1:.2f}) + {second_name} ({w2:.2f})")

        class BlendModel:
            def __init__(self, m1, m2, w1, w2):
                self.m1, self.m2, self.w1, self.w2 = m1, m2, w1, w2
            def predict(self, X):
                return self.w1 * self.m1.predict(X) + self.w2 * self.m2.predict(X)
            def fit(self, X, y):
                return self  # already fitted

        winner = BlendModel(m1, m2, w1, w2)
        best_name = blend_name
        best_rmse = first_rmse  # conservative: report the better of the two

        if verbose:
            print(f"  Winner: {best_name}  (RMSE≈{best_rmse:.3f})")
    else:
        winner = candidates[best_name]
        winner.fit(X_train, y_train)
        if verbose:
            print(f"\n  Winner: {best_name}  (RMSE={best_rmse:.3f} ±{best_std:.3f})")

    # --- NEW: compute noise ratio for spread decisions ---
    noise_ratio = best_rmse / profile.target_std if profile.target_std > 0 else 1.0
    profile.noise_ratio = round(noise_ratio, 4)

    # Flag no-quote only when model explains almost nothing
    profile.no_quote = noise_ratio > 0.98

    return {
        "name":       best_name,
        "model":      winner,
        "cv_rmse":    round(best_rmse, 4),
        "cv_std":     round(best_std, 4),
        "all_scores": {k: round(v[0], 4) for k, v in scores.items()},
        "noise_ratio": round(noise_ratio, 4),
        "blended":    use_blend,
    }


# Test
if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent))
    from data_check import profile_dataset
    
    sys.stdout = open("output.log", "w")

    DATA_DIR = Path(__file__).parent.parent / "hackathon_data"

    for i in range(1, 10):
        print(f"\n{'='*45}")
        print(f"  STOCK {i}")
        print(f"{'='*45}")
        train = pd.read_csv(DATA_DIR / f"stock_{i}_train.csv")
        test  = pd.read_csv(DATA_DIR / f"stock_{i}_test.csv")
        X = train.drop("target", axis=1)
        y = train["target"]
        profile = profile_dataset(train, test)
        result  = select_model(profile, X, y)
        pred    = result["model"].predict(test)[0]
        print(f"  Prediction: {pred:.2f}")
        print(f"  Noise ratio: {result['noise_ratio']:.2f}  |  No-quote: {profile.no_quote}")
    
    sys.stdout.close()