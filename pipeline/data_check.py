from dataclasses import dataclass, field
from typing import List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist


@dataclass
class DataProfile:
    n_train:          int   = 0             # Number of training rows
    n_features:       int   = 0             # Number of features (excluding target)
    missing_values:   int   = 0             # Total count of missing values in train + test
    has_missing:      bool  = False         # Whether there are any missing values    
    target_mean:      float = 0.0           # Mean of the target variable
    target_std:       float = 0.0           # Standard deviation of the target variable
    target_min:       float = 0.0           # Minimum value of the target variable
    target_max:       float = 0.0           # Maximum value of the target variable
    clip_low:         int   = 0             # Number of values clipped at the lower end
    clip_high:        int   = 0             # Number of values clipped at the upper end
    is_clipped:       bool  = False         # Whether the target variable is clipped
    n_outliers:       int   = 0             # Number of outliers in the target variable
    outlier_pct:      float = 0.0           # Percentage of outliers in the target variable
    heavy_tails:      bool  = False         # Whether the target variable has heavy tails
    already_scaled:   bool  = False         # Whether the features are already scaled
    top_target_corr:  float = 0.0           # Correlation of the most correlated feature with the target
    mean_target_corr: float = 0.0           # Mean correlation of all features with the target
    max_feat_corr:    float = 0.0           # Maximum correlation between any two features
    high_collinearity: bool = False         # Whether there is high collinearity between features
    signal_strength:  str   = "low"         # Strength of the signal in the data
    test_dist_p10:    float = 0.0           # 10th percentile of distances from test points to nearest train points
    test_min_dist:    float = 0.0           # Nearest train point distance
    test_ood:         bool  = False         # Whether the test set is out-of-distribution compared to the training set
    try_poly_features:    bool  = False     # Whether it's worth trying polynomial features
    use_robust_loss:      bool  = False     # Whether to use robust loss functions to handle heavy tails
    ood_spread_multiplier: float = 1.0      # Multiplier to increase model complexity if test set is OOD (based on distance ratio)
    noise_ratio:      float = 0.0           # NEW: cv_rmse / target_std
    no_quote:         bool  = False         # signal too weak to quote
    warnings: List[str] = field(default_factory=list) # List of warnings or flags about the dataset


def profile_dataset(train: pd.DataFrame, test: pd.DataFrame, target_col: str = "target") -> DataProfile:
    # Create a profile of the dataset
    p = DataProfile()

    #Split the training data into features and target
    X = train.drop(columns=[target_col])
    y = train[target_col]

    # Get row(instance) and column(feature) counts
    p.n_train    = len(train)
    p.n_features = X.shape[1]

    # Count missing values in both train and test sets
    p.missing_values = int(X.isnull().sum().sum() + test.isnull().sum().sum())
    p.has_missing    = p.missing_values > 0

    # Basic target mean, std, min, max
    p.target_mean = float(y.mean())
    p.target_std  = float(y.std())
    p.target_min  = float(y.min())
    p.target_max  = float(y.max())

    # Check for clipping at the min and max values
    p.clip_low   = int((y == y.min()).sum())
    p.clip_high  = int((y == y.max()).sum())
    p.is_clipped = p.clip_low >= 1 and p.clip_high >= 1

    # Identify outliers using z-scores (threshold of 3)
    z_scores      = (y - y.mean()) / y.std()
    p.n_outliers  = int((z_scores.abs() > 3).sum())
    p.outlier_pct = p.n_outliers / p.n_train * 100
    p.heavy_tails = p.outlier_pct > 2.0

    # Check if features are already scaled (rough heuristic based on mean and std)
    feat_means       = float(X.mean().abs().mean())
    feat_stds        = float(X.std().mean())
    p.already_scaled = feat_means < 0.5 and 0.5 < feat_stds < 2.5

    # Correlation analysis
    corrs = X.corrwith(y).abs().sort_values(ascending=False)
    p.top_target_corr  = float(corrs.iloc[0])
    p.mean_target_corr = float(corrs.mean())

    # Feature collinearity check
    fc = X.corr().abs().values.copy()
    np.fill_diagonal(fc, 0)
    p.max_feat_corr     = float(fc.max())
    p.high_collinearity = p.max_feat_corr > 0.7

    # Determine signal strength based on top feature correlation with target
    if p.top_target_corr > 0.4:
        p.signal_strength = "high"
    elif p.top_target_corr > 0.15:
        p.signal_strength = "medium"
    else:
        p.signal_strength = "low"

    #Scale features
    sc = StandardScaler().fit(X)
    X_scaled    = sc.transform(X)
    test_scaled = sc.transform(test)
  
    # Compute distance from test points to nearest train points as a simple OOD signal
    dist_matrix    = cdist(test_scaled, X_scaled, metric="euclidean")
    min_dists      = dist_matrix.min(axis=1)           # nearest neighbour per test row
    p.test_min_dist = float(min_dists.mean())

    # Also keep centroid distance as a secondary signal
    centroid       = X_scaled.mean(axis=0)
    centroid_dists = np.linalg.norm(test_scaled - centroid, axis=1)
    p.test_dist_p10 = float(np.percentile(centroid_dists, 10))

    # Simple heuristic: if test points are on average more than ~4 units away from nearest train points in scaled space, flag as OOD
    p.test_ood = p.test_min_dist > 4.0

    # If test set is OOD, we can consider increasing model complexity (e.g. via regularisation parameters or tree depth) to try to capture more complex patterns — this multiplier can be used later in model selection
    if p.test_ood:
        p.ood_spread_multiplier = min(p.test_min_dist / 2.5, 2.5)

    # Based on the profile, decide whether to try polynomial features and/or robust loss functions
    p.try_poly_features = (
        p.signal_strength in ("high", "medium")
        and not p.high_collinearity
        and p.n_train >= 50
    )

    # If heavy tails are detected, we can consider using robust loss functions (like Huber or quantile loss) in tree-based models to reduce the influence of outliers
    p.use_robust_loss = p.heavy_tails

    return p
