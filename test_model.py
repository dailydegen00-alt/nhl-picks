import joblib, pandas as pd, numpy as np
from scipy.stats import norm

bundle = joblib.load('data/model_total.pkl')
df = pd.read_csv('data/training_data.csv')
df['total_goals'] = df['home_score'] + df['away_score']

season_map = bundle.get('season_map', {})
if season_map:
    df['season_num'] = df['season'].map(season_map)

df = df.dropna(subset=bundle['features'] + ['total_goals'])

# Use last 635 games as test set (same split as backtest)
test = df.iloc[-635:].copy()
test['pred'] = bundle['model'].predict(test[bundle['features']])
test['over_prob'] = 1 - norm.cdf(6.5, loc=test['pred'], scale=bundle['residual_std'])
test['pick_over'] = test['over_prob'] >= 0.5
test['conf'] = np.where(test['pick_over'], test['over_prob'], 1 - test['over_prob'])
test['actual_over'] = test['total_goals'] > 6.5
test['correct'] = test['pick_over'] == test['actual_over']

JUICE = 1.909
print("Results using saved model directly on test set (6.5 line):")
print(f"{'Band':<10} {'W-L':<10} {'Acc':>6} {'ROI':>8} {'Games':>6}")
print("-" * 45)
for lo, hi, lbl in [(0.50,0.55,'50-55%'),(0.55,0.60,'55-60%'),
                     (0.60,0.65,'60-65%'),(0.65,0.70,'65-70%'),(0.70,1.01,'70%+')]:
    sub = test[(test['conf'] >= lo) & (test['conf'] < hi)]
    if len(sub) == 0:
        continue
    w = int(sub['correct'].sum())
    l = len(sub) - w
    acc = sub['correct'].mean()
    roi = (w * (JUICE-1) * 100 - l * 100) / (len(sub) * 100) * 100
    flag = ' +' if roi > 0 else ' -'
    print(f"{lbl:<10} {w}-{l:<7} {acc:>6.1%} {roi:>+7.1f}%{flag}  ({len(sub)} games)")

print()
conf_mask = test['conf'] >= 0.60
sub = test[conf_mask]
if len(sub) > 0:
    w = int(sub['correct'].sum())
    l = len(sub) - w
    roi = (w * (JUICE-1) * 100 - l * 100) / (len(sub) * 100) * 100
    print(f"All 60%+: {w}-{l} ({sub['correct'].mean():.1%}) ROI: {roi:+.1f}% ({len(sub)} games)")
