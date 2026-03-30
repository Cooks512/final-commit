# QMML Market Making Hackathon — Pipeline

A four-component pipeline for the QMML Market Making Hackathon: data profiling, model selection, spread quoting, and live trade decisions.

## Quick Start

```bash
# Generate all quotes (pre-competition)
python pipeline/spread.py

# During live rounds — interactive trade helper
python pipeline/trade_decide.py

# Single stock trade decision
python pipeline/trade_decide.py --stock 3 --mm_bid 240 --mm_ask 260 --capital 95000
```

## Competition Rules (Summary)

- 9 rounds, each with a train set (many rows) and a test set (1 row).
- Each team submits a **bid** and **ask** price per round.
- The team with the **tightest spread becomes the Market Maker (MM)**.
- The MM **must accept all trades** from every other team.
- Other teams must trade at least **10 shares** against the MM.
- True price is revealed after trading. P&L is calculated.
- **Running out of money = elimination.**
- Prizes: highest total P&L, best Sortino ratio, best undergrad P&L.

## Strategic Insight

Being the Market Maker is **high risk**. Every other team trades against you using their own predictions. If your prediction is even slightly off, they all exploit the mispricing. The MM role is only profitable when your prediction is highly accurate and your spread is wide enough to absorb the error.

The pipeline therefore operates in two modes:

| Mode | When | Spread | Goal |
|------|------|--------|------|
| **ATTACK** | noise_ratio < 0.25, overfit_ratio > 0.90, RMSE < 5% of price | Tight (~3× CV RMSE) | Win MM role and profit from the spread |
| **SURVIVE** | Everything else | Wide (0.5–2× target σ) | Avoid MM role; profit by trading against whoever is MM |

---

## Pipeline Architecture

```
hackathon_data/
  stock_{1-9}_train.csv
  stock_{1-9}_test.csv

pipeline/
  data_check.py      ← Data profiling & diagnostics
  model_select.py    ← Model selection & CV
  spread.py          ← Bid/ask spread computation
  trade_decide.py    ← Live trade decision helper
```

**Data flow:**

```
train.csv + test.csv
       │
       ▼
  data_check.py  →  DataProfile
       │
       ▼
  model_select.py  →  prediction + cv_rmse + noise_ratio
       │
       ▼
  spread.py  →  bid / ask / spread / role
       │
       ▼
  trade_decide.py  →  buy/sell decision + share sizing (live)
```

---

## Component 1: `data_check.py`

**Purpose:** Profile each stock's dataset to inform model selection and spread sizing.

**Input:** Train DataFrame, test DataFrame.

**Output:** A `DataProfile` dataclass with ~25 fields.

### Key diagnostics computed

**Target statistics** — mean, std, min, max of the target column. These are used downstream as anchors (mean-reversion centre) and scaling factors (spread ceiling).

**Signal strength** — computed from the highest absolute Pearson correlation between any feature and the target.

- `top_target_corr > 0.4` → **high** signal
- `top_target_corr > 0.15` → **medium** signal
- Otherwise → **low** signal

This directly drives the aggression level in spread.py. High signal = tighter spread.

**Collinearity check** — maximum pairwise correlation between features. If > 0.7, polynomial features are skipped (they'd explode dimensionality without adding information).

**Outlier detection** — z-score based. If > 2% of targets have |z| > 3, the dataset is flagged as heavy-tailed and `use_robust_loss` is set.

**Out-of-distribution (OOD) detection** — the test point's distance to its nearest training neighbour in standardised feature space, using `scipy.spatial.distance.cdist`. If the mean nearest-neighbour distance exceeds 4.0 standard deviations, the test point is flagged as OOD and the spread is widened by `ood_spread_multiplier` (up to 2.5×).

**Polynomial feature gate** — `try_poly_features` is set when signal is high/medium, collinearity is low, and n_train ≥ 50. This prevents model_select from wasting CV time on polynomial features for noisy or small datasets.

### Fields populated later by model_select

- `noise_ratio` — cv_rmse / target_std. Measures how much of the target's variance the model explains. Near 0 = strong signal. Near 1 = model is barely better than predicting the mean.
- `no_quote` — legacy flag (no longer used in the current spread strategy, since all stocks must be quoted).

---

## Component 2: `model_select.py`

**Purpose:** Run cross-validated model selection across multiple candidates, optionally blend top models, and return the best model with uncertainty estimates.

**Input:** DataProfile, X_train, y_train.

**Output:** Dict with model name, fitted model, cv_rmse, cv_std, noise_ratio, all individual model scores.

### Candidate models

Always included:

| Model | Why |
|-------|-----|
| **Ridge** (with StandardScaler) | Strong baseline for small/noisy datasets. Alphas searched over 7 orders of magnitude. |
| **BayesianRidge** (with StandardScaler) | Similar to Ridge but provides uncertainty estimates. Wrapped in a scaling pipeline for fair comparison. |
| **ElasticNet** (with StandardScaler) | L1+L2 regularisation. Useful under collinearity because L1 can zero out redundant features. Searches over 5 l1_ratios × 5 alphas. |

Conditionally included:

| Model | Condition | Why conditional |
|-------|-----------|-----------------|
| **Ridge+Poly** (degree 2) | `profile.try_poly_features` is True | Polynomial features explode dimensionality. Only useful when signal is medium/high, collinearity is low, and dataset is large enough. |
| **LightGBM** | `n_train ≥ 80` | Tree models overfit badly on small datasets. Tuned conservatively: 200 estimators, 15 leaves, strong L1/L2 regularisation, min_child_samples adaptive to dataset size. |
| **XGBoost** | `n_train ≥ 80` | Same reasoning. max_depth capped at 3, strong regularisation. |

### Cross-validation setup

- K-fold CV with `n_folds = min(5, max(3, n_train // 10))`.
- Scoring: `neg_root_mean_squared_error`.
- Fixed `random_state=42` for reproducibility.

### Blending logic

After CV, the top-2 models are compared. If:

1. Their RMSE gap is < 1.5%, **AND**
2. The noise_ratio (cv_rmse / target_std) is < 0.7

...they are blended with weights inversely proportional to their RMSE. The blend is implemented as a lightweight `BlendModel` class that averages predictions.

**Why the gates:** Blending two bad models doesn't help. The noise_ratio gate ensures blending only happens when the models have genuine signal. The gap gate ensures blending only happens when the models genuinely disagree (otherwise you're just averaging two nearly identical predictions).

### Noise ratio

After selecting the winner, model_select computes:

```
noise_ratio = cv_rmse / target_std
```

This is the single most important number in the pipeline. It determines:
- Whether spread.py goes ATTACK or SURVIVE
- How aggressive the spread is
- Whether the prediction is mean-anchored
- How trade_decide.py sizes positions

---

## Component 3: `spread.py`

**Purpose:** Compute bid/ask quotes for each stock, informed by game theory.

**Input:** Prediction, cv_rmse, DataProfile, overfit_ratio.

**Output:** Dict with prediction, bid, ask, spread, aggression, role (ATTACK/SURVIVE).

### Step 1: Role decision

```python
want_mm = (
    noise_ratio < 0.25        # strong signal
    and overfit_ratio > 0.90   # not overfitting
    and cv_rmse < price * 0.05 # RMSE < 5% of predicted price
)
```

All three conditions must be true. In the current dataset, only stocks 1 and 2 qualify.

### Step 2: Set aggression

**ATTACK mode** (want_mm = True):
- `aggression = 0.7`
- Goal: tight spread to win the MM seat, but not suicidally tight.

**SURVIVE mode** (want_mm = False):
- Aggression scales down with noise ratio:
  - noise < 0.5 → `aggression = 0.15` (moderately wide)
  - noise < 0.8 → `aggression = 0.08` (quite wide)
  - noise ≥ 0.8 → `aggression = 0.03` (very wide, just participating)

### Step 3: Compute half-spread

```python
multiplier  = 2.0 - 1.5 * aggression
half_spread = cv_rmse * multiplier
```

At aggression 0.7 (ATTACK), the multiplier is 0.95, so half-spread ≈ 1× RMSE each side.
At aggression 0.03 (max SURVIVE), the multiplier is 1.955, so half-spread ≈ 2× RMSE each side.

**Overfit penalty:** If `overfit_ratio < 0.85` (train error suspiciously low vs CV error), the spread is widened by `1 + (0.85 - ratio)`. For example, a 0.75 ratio adds a 10% penalty.

**OOD penalty:** Multiplied by `ood_spread_multiplier` from data_check (1.0 if not OOD, up to 2.5 if test point is far from training distribution).

### Step 4: Bounds

- **ATTACK:** Half-spread clipped to `[1.5× RMSE, 3× RMSE]`. Floor ensures you can survive being MM. Ceiling prevents unnecessarily wide quotes that lose the MM seat.
- **SURVIVE:** Half-spread clipped to `[0.5× target_std, 2× target_std]`. High floor ensures you never accidentally submit a competitive spread.

### Step 5: Mean-anchoring

When `noise_ratio > 0.8`, the prediction is blended toward `target_mean`:

```python
mean_weight = min((noise - 0.8) / 0.2, 0.6)
prediction = prediction * (1 - mean_weight) + target_mean * mean_weight
```

At noise_ratio = 1.0, the prediction is 60% model / 40% mean. Rationale: if the model barely beats predicting the mean, the mean is a safer centre for your quotes.

### Step 6: Asymmetric spread

A mild mean-reversion bias shifts the bid and ask asymmetrically:

```python
deviation = (prediction - target_mean) / target_std
asym = clip(deviation * 0.1, -0.2, 0.2)
bid_half = half_spread * (1 + asym)   # wider below if pred < mean
ask_half = half_spread * (1 - asym)   # wider above if pred > mean
```

If the prediction is above the mean, the ask is slightly tighter (we expect mean-reversion downward) and the bid is slightly wider. This gives a small edge on stocks with mean-reverting targets.

---

## Component 4: `trade_decide.py`

**Purpose:** During live rounds, decide whether to buy or sell against the Market Maker and how many shares.

**Input:** Your prediction, CV RMSE, MM's bid/ask, your current capital.

**Output:** Action (BUY/SELL), number of shares, reasoning.

### Usage modes

**Interactive mode** (run with no args):
```bash
python pipeline/trade_decide.py
```
Walks through all 9 rounds. For each stock, shows your prediction and noise ratio, then prompts for the MM's bid/ask. After the true price is revealed, tracks your running P&L.

**Single stock mode:**
```bash
python pipeline/trade_decide.py --stock 3 --mm_bid 240 --mm_ask 260 --capital 95000
```

### Decision logic

**Direction:**
- `prediction > mm_ask` → **BUY** (you think it's worth more than the MM is selling for)
- `prediction < mm_bid` → **SELL** (you think it's worth less than the MM is buying at)
- Prediction inside spread → trade minimum 10 shares, pick the side closer to your prediction

**Edge measurement:**
- `edge` = absolute difference between prediction and trade price
- `edge_zscore` = edge / cv_rmse (how many "standard errors" of edge you have)

**Position sizing:**

| Edge (σ) | Sizing |
|----------|--------|
| ≤ 0 | 10 shares (minimum) |
| 0 – 0.5 | 15 shares |
| 0.5 – 1.0 | 30% × confidence × max_shares |
| 1.0 – 2.0 | 50% × confidence × max_shares |
| > 2.0 | 70% × confidence × max_shares |

Where:
- `confidence = max(0.05, 1 - noise_ratio)`
- `max_shares = (capital × 15%) / trade_price`

**Survival check:** If worst-case loss (shares × cv_rmse) exceeds 25% of capital, position is scaled back to the minimum 10 shares regardless of edge.

---

## Current Output (All 9 Stocks)

```
Stock  Model                           pred     bid     ask  spread   aggr  noise  ofit     role
---------------------------------------------------------------------------------------------------------
1      Blend(ElasticNet+Ridge)       273.93  266.11  280.71   14.60  0.700   0.12  1.00   ATTACK
2      Blend(Ridge+BayesianRidge)    220.91  206.28  235.44   29.16  0.700   0.20  0.99   ATTACK
3      Blend(BayesianRidge+ElasticNet)  259.24  203.74  307.66  103.92  0.150   0.34  0.74  SURVIVE
4      XGBoost                       239.95  213.00  268.54   55.54  0.030   0.88  0.95  SURVIVE
5      Ridge                         250.63  219.45  281.03   61.57  0.030   0.91  0.99  SURVIVE
6      ElasticNet                    172.40  117.46  227.30  109.84  0.030   0.96  1.03  SURVIVE
7      XGBoost                       212.93  196.80  228.48   31.68  0.030   0.98  0.96  SURVIVE
8      Ridge                         199.66  173.00  225.63   52.63  0.030   0.99  0.99  SURVIVE
9      BayesianRidge                 218.61  174.86  262.35   87.49  0.030   0.95  1.05  SURVIVE
```

---

## Dependencies

```
numpy
pandas
scikit-learn
lightgbm
xgboost
scipy
```

---

## Key Metrics Glossary

| Metric | Formula | Meaning |
|--------|---------|---------|
| **noise_ratio** | cv_rmse / target_std | 0 = perfect model, 1 = no better than predicting the mean |
| **overfit_ratio** | train_rmse / cv_rmse | 1.0 = no overfit, < 0.85 = significant overfit |
| **aggression** | 0–1 scale | Higher = tighter spread = more likely to become MM |
| **edge_zscore** | edge / cv_rmse | How many standard errors of predicted profit you have |
| **ood_spread_multiplier** | 1.0–2.5 | Spread widening factor when test point is far from training data |
fix attribution
