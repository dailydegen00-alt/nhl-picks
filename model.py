import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib, os

df = pd.read_csv('data/training_data.csv')
df = df.sort_values('date').reset_index(drop=True)

# ── All candidate features ─────────────────────────────────────────────────────
candidates = [
    # Season-level possession stats
    'xg_pct_diff', 'corsi_pct_diff', 'fenwick_pct_diff',
    'sv_xg_pct_diff', 'hdc_pct_diff', 'pdo_diff',
    'pp_threat_diff', 'pk_exposure_diff',
    'home_xg_pct', 'away_xg_pct',
    'home_sv_xg_pct', 'away_sv_xg_pct',
    'home_hdc_pct', 'away_hdc_pct',
    'home_pdo', 'away_pdo',
    'home_pp_threat', 'away_pp_threat',
    # Goalie quality
    'home_goalie_gsax', 'away_goalie_gsax', 'gsax_diff',
    # Season context
    'season_phase', 'season_progress', 'is_playoffs',
    # Rolling form — L10 (existing)
    'form_diff', 'gpg_diff',
    # Rolling form — NEW
    'gapg_diff',          # goals against pg differential (defensive form)
    'gdiff_l10_diff',     # goal differential L10
    'form_l5_diff',       # win rate last 5 games
    'gdiff_l5_diff',      # goal diff last 5 games
    'home_form_l5', 'away_form_l5',
    'home_gapg_l10', 'away_gapg_l10',
    'home_gdiff_l10', 'away_gdiff_l10',
]
candidates = [f for f in candidates if f in df.columns]

missing = [f for f in ['home_goalie_gsax','season_phase','form_l5_diff','gapg_diff']
           if f not in df.columns]
if missing:
    print(f"WARNING: missing features (run data prep scripts): {missing}")

df = df.dropna(subset=candidates + ['home_win'])
X_all = df[candidates]
y     = df['home_win']

print(f"Training on {len(df)} games | {len(candidates)} candidates\n")

# ── Test each feature individually ────────────────────────────────────────────
print("=== Individual feature power ===")
pipe = Pipeline([('s', StandardScaler()), ('m', LogisticRegression(max_iter=1000))])
results = []
for feat in candidates:
    scores = cross_val_score(pipe, df[[feat]], y, cv=5)
    results.append((feat, scores.mean()))
    print(f"  {feat:<28} {scores.mean():.1%}")

results.sort(key=lambda x: x[1], reverse=True)
print(f"\nTop 12 features:")
for feat, score in results[:12]:
    print(f"  {feat:<28} {score:.1%}")

top_features = [r[0] for r in results[:12]]
df2 = df.dropna(subset=top_features + ['home_win'])
X   = df2[top_features]
y2  = df2['home_win']

# ── Regularization tuning ─────────────────────────────────────────────────────
print(f"\n=== Regularization tuning (C value) ===")
best_c = 1.0
best_score = 0.0
tscv = TimeSeriesSplit(n_splits=5)

for C in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
    pipe_c = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  LogisticRegression(max_iter=1000, C=C))
    ])
    scores = cross_val_score(pipe_c, X, y2, cv=tscv)
    avg = scores.mean()
    marker = ' ← best' if avg > best_score else ''
    print(f"  C={C:<6} {avg:.1%}{marker}")
    if avg > best_score:
        best_score = avg
        best_c = C

print(f"\nBest C: {best_c} → {best_score:.1%}")

# ── Vig-aware sample weighting ────────────────────────────────────────────────
sample_weights = None
if os.path.exists('data/historical_ml_odds.csv'):
    print(f"\n=== Vig-aware weighting ===")
    odds_df = pd.read_csv('data/historical_ml_odds.csv')

    FULL_TO_ABBR = {
        'Anaheim Ducks':'ANA','Boston Bruins':'BOS','Buffalo Sabres':'BUF',
        'Calgary Flames':'CGY','Carolina Hurricanes':'CAR','Chicago Blackhawks':'CHI',
        'Colorado Avalanche':'COL','Columbus Blue Jackets':'CBJ','Dallas Stars':'DAL',
        'Detroit Red Wings':'DET','Edmonton Oilers':'EDM','Florida Panthers':'FLA',
        'Los Angeles Kings':'LAK','Minnesota Wild':'MIN','Montréal Canadiens':'MTL',
        'Montreal Canadiens':'MTL','Nashville Predators':'NSH','New Jersey Devils':'NJD',
        'New York Rangers':'NYR','NY Rangers':'NYR','New York Islanders':'NYI',
        'NY Islanders':'NYI','Ottawa Senators':'OTT','Philadelphia Flyers':'PHI',
        'Pittsburgh Penguins':'PIT','San Jose Sharks':'SJS','Seattle Kraken':'SEA',
        'St. Louis Blues':'STL','Tampa Bay Lightning':'TBL','Toronto Maple Leafs':'TOR',
        'Utah Mammoth':'UTA','Vancouver Canucks':'VAN','Vegas Golden Knights':'VGK',
        'Washington Capitals':'WSH','Winnipeg Jets':'WPG',
    }
    def to_abbr(t):
        return FULL_TO_ABBR.get(str(t).strip(), str(t).strip())

    odds_map = {}
    for _, r in odds_df.iterrows():
        odds_map[(r['date'], r['home_team'], r['away_team'])] = {
            'home_imp': float(r['home_implied']) / 100,
            'away_imp': float(r['away_implied']) / 100,
        }

    # Weight = how sharp/confident the market is
    # High vig (one-sided market) = sharp game = higher weight
    # Calculated as: market confidence = max(home_imp, away_imp)
    # Games where market strongly favors one side = more informative
    weights = []
    matched = 0
    for _, row in df2.iterrows():
        ha = to_abbr(row['home_team'])
        aa = to_abbr(row['away_team'])
        key = (str(row['date'])[:10], ha, aa)
        o = odds_map.get(key)
        if o:
            confidence = max(o['home_imp'], o['away_imp'])
            weights.append(0.5 + confidence)  # range ~1.0–1.45
            matched += 1
        else:
            weights.append(1.0)  # neutral weight for unmatched games
    sample_weights = np.array(weights)
    print(f"  Matched {matched}/{len(df2)} games to real odds for weighting")
    print(f"  Weight range: {sample_weights.min():.2f} – {sample_weights.max():.2f}")

    # Test vig-weighted vs unweighted
    pipe_w = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  LogisticRegression(max_iter=1000, C=best_c))
    ])
    scores_w = []
    scores_u = []
    for train_idx, test_idx in tscv.split(X):
        sw = sample_weights[train_idx] if sample_weights is not None else None
        pipe_w.fit(X.iloc[train_idx], y2.iloc[train_idx],
                   model__sample_weight=sw)
        scores_w.append(accuracy_score(y2.iloc[test_idx],
                                        pipe_w.predict(X.iloc[test_idx])))
        pipe_w.fit(X.iloc[train_idx], y2.iloc[train_idx])
        scores_u.append(accuracy_score(y2.iloc[test_idx],
                                        pipe_w.predict(X.iloc[test_idx])))

    print(f"  Weighted accuracy:   {np.mean(scores_w):.1%}")
    print(f"  Unweighted accuracy: {np.mean(scores_u):.1%}")
    use_weights = np.mean(scores_w) > np.mean(scores_u)
    print(f"  Using weights: {use_weights}")
    if not use_weights:
        sample_weights = None
else:
    print(f"\nNo historical_ml_odds.csv — skipping vig weighting")

# ── Final model ────────────────────────────────────────────────────────────────
model = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LogisticRegression(max_iter=1000, C=best_c))
])

scores = []
for train_idx, test_idx in tscv.split(X):
    sw = sample_weights[train_idx] if sample_weights is not None else None
    model.fit(X.iloc[train_idx], y2.iloc[train_idx], model__sample_weight=sw)
    preds = model.predict(X.iloc[test_idx])
    scores.append(accuracy_score(y2.iloc[test_idx], preds))

avg = sum(scores)/len(scores)
print(f"\n=== Final model ===")
print(f"Baseline:      57.3%")
print(f"Previous best: 62.3%")
print(f"Our model:     {avg:.1%}  (C={best_c}, weighted={sample_weights is not None})")
print(f"\n{'✅ NEW BEST!' if avg > 0.623 else ('✅ BEAT BASELINE' if avg > 0.573 else '❌ Below baseline')}")

# Fit on all data
model.fit(X, y2, model__sample_weight=sample_weights)
joblib.dump({
    'model':    model,
    'features': top_features,
    'best_c':   best_c,
    'weighted': sample_weights is not None,
}, 'data/model.pkl')
print(f"\n💾 Saved — {len(df2)} games")
print(f"Features: {top_features}")