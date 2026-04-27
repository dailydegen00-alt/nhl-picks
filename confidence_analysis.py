import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import numpy as np

df = pd.read_csv('data/training_data.csv')
df = df.sort_values('date').reset_index(drop=True)

features = [
    'home_xg_pct',      'away_xg_pct',      'xg_pct_diff',
    'home_corsi_pct',   'away_corsi_pct',   'corsi_pct_diff',
    'home_fenwick_pct', 'away_fenwick_pct', 'fenwick_pct_diff',
    'home_sv_xg_pct',   'away_sv_xg_pct',   'sv_xg_pct_diff',
    'home_hdc_pct',     'away_hdc_pct',     'hdc_pct_diff',
    'home_pdo',         'away_pdo',         'pdo_diff',
    'form_diff',        'gpg_diff',
]
features = [f for f in features if f in df.columns]
df = df.dropna(subset=features + ['home_win'])

X = df[features]
y = df['home_win']

# Collect all out-of-fold predictions
tscv = TimeSeriesSplit(n_splits=5)
model = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LogisticRegression(max_iter=1000))
])

all_probs  = np.zeros(len(df))
all_actual = np.zeros(len(df))

for train_idx, test_idx in tscv.split(X):
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    probs = model.predict_proba(X.iloc[test_idx])[:,1]
    all_probs[test_idx]  = probs
    all_actual[test_idx] = y.iloc[test_idx].values

# Max confidence = distance from 50%
all_conf = np.maximum(all_probs, 1 - all_probs)
all_picks = (all_probs >= 0.5).astype(int)
correct   = (all_picks == all_actual).astype(int)

results = pd.DataFrame({
    'prob':    all_probs,
    'conf':    all_conf,
    'correct': correct,
    'actual':  all_actual,
})

print("=== Accuracy by confidence band ===\n")
bands = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 1.00)]
for lo, hi in bands:
    mask = (results['conf'] >= lo) & (results['conf'] < hi)
    subset = results[mask]
    if len(subset) == 0:
        continue
    acc = subset['correct'].mean()
    n   = len(subset)
    pct_of_games = n / len(results) * 100
    print(f"  {lo:.0%}–{hi:.0%} confidence: {acc:.1%} accuracy  "
          f"({n} games, {pct_of_games:.0f}% of all games)")

print()
print("=== If we only bet 60%+ confidence ===")
high_conf = results[results['conf'] >= 0.60]
print(f"  Games:    {len(high_conf)} ({len(high_conf)/len(results)*100:.0f}% of slate)")
print(f"  Accuracy: {high_conf['correct'].mean():.1%}")

print()
print("=== If we only bet 65%+ confidence ===")
very_high = results[results['conf'] >= 0.65]
print(f"  Games:    {len(very_high)} ({len(very_high)/len(results)*100:.0f}% of slate)")
if len(very_high) > 0:
    print(f"  Accuracy: {very_high['correct'].mean():.1%}")

print()
print("=== Overall ===")
print(f"  All games accuracy: {results['correct'].mean():.1%}")
print(f"  Avg confidence:     {results['conf'].mean():.1%}")
