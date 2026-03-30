from dataclasses import dataclass, field
from typing import List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class DataProfile:
    n_train:          int   = 0
    n_features:       int   = 0
    missing_values:   int   = 0
    has_missing:      bool  = False
    target_mean:      float = 0.0
    target_std:       float = 0.0
    target_min:       float = 0.0
    target_max:       float = 0.0
    clip_low:         int   = 0
    clip_high:        int   = 0
    is_clipped:       bool  = False
    n_outliers:       int   = 0
    outlier_pct:      float = 0.0
    heavy_tails:      bool  = False
    already_scaled:   bool  = False
    top_target_corr:  float = 0.0
    mean_target_corr: float = 0.0
    max_feat_corr:    float = 0.0
    high_collinearity: bool = False
    signal_strength:  str   = "low"
    test_dist_p10:    float = 0.0
    test_min_dist:    float = 0.0          # NEW: closest train point distance
    test_ood:         bool  = False
    try_poly_features:    bool  = False
    use_robust_loss:      bool  = False
    ood_spread_multiplier: float = 1.0
    noise_ratio:      float = 0.0          # NEW: cv_rmse / target_std
    no_quote:         bool  = False        # NEW: signal too weak to quote
    warnings: List[str] = field(default_factory=list)


def profile_dataset(train: pd.DataFrame, test: pd.DataFrame,
                    target_col: str = "target") -> DataProfile:
    p = DataProfile()
    X = train.drop(columns=[target_col])
    y = train[target_col]

    p.n_train    = len(train)
    p.n_features = X.shape[1]

    p.missing_values = int(X.isnull().sum().sum() + test.isnull().sum().sum())
    p.has_missing    = p.missing_values > 0

    p.target_mean = float(y.mean())
    p.target_std  = float(y.std())
    p.target_min  = float(y.min())
    p.target_max  = float(y.max())

    p.clip_low   = int((y == y.min()).sum())
    p.clip_high  = int((y == y.max()).sum())
    p.is_clipped = p.clip_low >= 1 and p.clip_high >= 1

    z_scores      = (y - y.mean()) / y.std()
    p.n_outliers  = int((z_scores.abs() > 3).sum())
    p.outlier_pct = round(p.n_outliers / p.n_train * 100, 2)
    p.heavy_tails = p.outlier_pct > 2.0

    feat_means       = float(X.mean().abs().mean())
    feat_stds        = float(X.std().mean())
    p.already_scaled = feat_means < 0.5 and 0.5 < feat_stds < 2.5

    corrs = X.corrwith(y).abs().sort_values(ascending=False)
    p.top_target_corr  = round(float(corrs.iloc[0]), 4)
    p.mean_target_corr = round(float(corrs.mean()), 4)

    fc = X.corr().abs().values.copy()
    np.fill_diagonal(fc, 0)
    p.max_feat_corr     = round(float(fc.max()), 4)
    p.high_collinearity = p.max_feat_corr > 0.7

    if p.top_target_corr > 0.4:
        p.signal_strength = "high"
    elif p.top_target_corr > 0.15:
        p.signal_strength = "medium"
    else:
        p.signal_strength = "low"

    # --- FIX: proper per-test-point OOD detection ---
    sc = StandardScaler().fit(X)
    X_scaled    = sc.transform(X)
    test_scaled = sc.transform(test)
    # Distance from each test point to its nearest training point
    # For single test point this is exact; for multiple it's still correct
    from scipy.spatial.distance import cdist
    dist_matrix    = cdist(test_scaled, X_scaled, metric="euclidean")
    min_dists      = dist_matrix.min(axis=1)           # nearest neighbour per test row
    p.test_min_dist = round(float(min_dists.mean()), 4)
    # Also keep centroid distance as a secondary signal
    centroid       = X_scaled.mean(axis=0)
    centroid_dists = np.linalg.norm(test_scaled - centroid, axis=1)
    p.test_dist_p10 = round(float(np.percentile(centroid_dists, 10)), 4)

    p.test_ood = p.test_min_dist > 4.0

    if p.test_ood:
        p.ood_spread_multiplier = round(min(p.test_min_dist / 2.5, 2.5), 2)

    p.try_poly_features = (
        p.signal_strength in ("high", "medium")
        and not p.high_collinearity
        and p.n_train >= 50
    )
    p.use_robust_loss = p.heavy_tails

    return p