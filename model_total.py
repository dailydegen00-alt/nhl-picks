# model_total.py — Regression approach for O/U predictions
# Instead of classifying over/under a fixed line, we predict actual total goals.
# At prediction time, compare predicted total to tonight's book line.
# Confidence comes from the residual distribution — no more calibration hacks.

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import norm
import joblib

# ── Load training data ────────────────────────────────────────────────────────
df = pd.read_csv('data/training_data.csv')

# Target: total goals in the game
df['total_goals'] = df['home_score'] + df['away_score']
df = df.dropna(subset=['total_goals'])

print(f"Total games loaded: {len(df)}")
print(f"Mean total goals: {df['total_goals'].mean():.2f}")
print(f"Std  total goals: {df['total_goals'].std():.2f}")
print(f"\nGoal total distribution:")
print(df['total_goals'].value_counts().sort_index().to_string())

# ── Features ──────────────────────────────────────────────────────────────────
# These mirror what's available in build_html.py via MoneyPuck team stats.
# All are per-game rates so they're scale-stable across seasons.
FEATURES = [
    'home_sv_xgf',  'home_sv_xga',   # score-adjusted xG for/against
    'away_sv_xgf',  'away_sv_xga',
    'home_hdcf',    'home_hdca',      # high-danger chances for/against
    'away_hdcf',    'away_hdca',
    'home_pdo',     'away_pdo',       # luck indicator — will regress
]

df = df.dropna(subset=FEATURES)
print(f"\nGames after dropping missing features: {len(df)}")

X = df[FEATURES]
y = df['total_goals']

# ── Model comparison ──────────────────────────────────────────────────────────
candidates = {
    'Ridge': Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=1.0))
    ]),
    'GBM': GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    ),
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
print("\n── Cross-validation (MAE) ──")

results = {}
for name, model in candidates.items():
    scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error')
    mae = -scores.mean()
    std = scores.std()
    results[name] = mae
    print(f"  {name:8s}: MAE = {mae:.4f} ± {std:.4f}")

best_name = min(results, key=results.get)
best_model = candidates[best_name]
print(f"\nBest model: {best_name} (MAE: {results[best_name]:.4f})")

# ── Fit on full dataset ───────────────────────────────────────────────────────
best_model.fit(X, y)

# Compute residual std — used at prediction time to turn margin-vs-line
# into a real probability via the normal CDF.
train_preds = best_model.predict(X)
residuals = y - train_preds
residual_std = float(residuals.std())

print(f"\nTrain residual std (used for confidence): {residual_std:.4f}")
print(f"  → A book line 0.5 goals from predicted = "
      f"{(1 - norm.cdf(0, loc=0.5, scale=residual_std))*100:.1f}% confidence")
print(f"  → A book line 1.0 goals from predicted = "
      f"{(1 - norm.cdf(0, loc=1.0, scale=residual_std))*100:.1f}% confidence")

# ── Sanity check: over/under split at 5.5 ────────────────────────────────────
# Tells you if the model is reasonably balanced (should be ~50/50)
prob_over_5_5 = [1 - norm.cdf(5.5, loc=p, scale=residual_std) for p in train_preds]
pct_over = sum(p > 0.5 for p in prob_over_5_5) / len(prob_over_5_5)
print(f"\nSanity check — % predicted OVER at 5.5 line: {pct_over*100:.1f}%  (want ~50%)")

# ── Save ──────────────────────────────────────────────────────────────────────
bundle = {
    'model':        best_model,
    'features':     FEATURES,
    'residual_std': residual_std,
    'model_name':   best_name,
}
joblib.dump(bundle, 'data/model_total.pkl')
print(f"\n✓ Saved data/model_total.pkl")
print("  Keys: model, features, residual_std, model_name")