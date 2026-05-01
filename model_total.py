# model_total.py — O/U regression model
# Features: 10 shot quality + 8 form + gsax_diff
# OOF residual std for honest confidence calibration

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import norm
import joblib

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv('data/training_data.csv')
df['total_goals'] = df['home_score'] + df['away_score']
df = df.dropna(subset=['total_goals'])

print(f"Total games loaded: {len(df)}")
print(f"Mean total goals: {df['total_goals'].mean():.2f}")
print(f"Std  total goals: {df['total_goals'].std():.2f}")

# ── Features ──────────────────────────────────────────────────────────────────
SHOT_QUALITY = [
    'home_sv_xgf',  'home_sv_xga',
    'away_sv_xgf',  'away_sv_xga',
    'home_hdcf',    'home_hdca',
    'away_hdcf',    'away_hdca',
    'home_pdo',     'away_pdo',
]

FORM = [
    'home_gpg_l10', 'away_gpg_l10',
    'home_gapg_l10','away_gapg_l10',
    'home_gpg_l5',  'away_gpg_l5',
    'home_gdiff_l10','away_gdiff_l10',
]

GOALIE = [
    'gsax_diff',  # positive = home goalie better, suppresses scoring
]

FEATURES = SHOT_QUALITY + FORM + GOALIE
FEATURES = [f for f in FEATURES if f in df.columns]
print(f"\nUsing {len(FEATURES)} features: {FEATURES}")

df = df.dropna(subset=FEATURES)
print(f"Games after dropping missing: {len(df)}")

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
    results[name] = mae
    print(f"  {name:8s}: MAE = {mae:.4f} +/- {scores.std():.4f}")

best_name = min(results, key=results.get)
best_model = candidates[best_name]
print(f"\nBest model: {best_name} (MAE: {results[best_name]:.4f})")

# ── Fit on full dataset ───────────────────────────────────────────────────────
best_model.fit(X, y)

if best_name == 'GBM':
    feat_imp = sorted(zip(FEATURES, best_model.feature_importances_), key=lambda x: -x[1])
    print("\nTop feature importances:")
    for feat, imp in feat_imp:
        print(f"  {feat:25s}: {imp:.4f}")

# ── OOF residual std (honest calibration) ────────────────────────────────────
print("\nComputing OOF residual std...")
oof_preds = np.zeros(len(y))
for train_idx, val_idx in kf.split(X):
    if best_name == 'GBM':
        m = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                       learning_rate=0.05, subsample=0.8, random_state=42)
    else:
        m = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=1.0))])
    m.fit(X.iloc[train_idx], y.iloc[train_idx])
    oof_preds[val_idx] = m.predict(X.iloc[val_idx])

oof_residual_std = float(np.std(y.values - oof_preds))
train_residual_std = float(np.std(y.values - best_model.predict(X)))
print(f"  Train residual std (optimistic): {train_residual_std:.4f}")
print(f"  OOF residual std   (honest):     {oof_residual_std:.4f}")

print(f"\n  -> Book line 0.5 from predicted = "
      f"{(1 - norm.cdf(0, loc=0.5, scale=oof_residual_std))*100:.1f}% confidence")
print(f"  -> Book line 1.0 from predicted = "
      f"{(1 - norm.cdf(0, loc=1.0, scale=oof_residual_std))*100:.1f}% confidence")

# ── Save ──────────────────────────────────────────────────────────────────────
bundle = {
    'model':        best_model,
    'features':     FEATURES,
    'residual_std': oof_residual_std,
    'model_name':   best_name,
}
joblib.dump(bundle, 'data/model_total.pkl')
print(f"\nSaved data/model_total.pkl")
print(f"  Model: {best_name} | Features: {len(FEATURES)} | Residual std: {oof_residual_std:.4f}")