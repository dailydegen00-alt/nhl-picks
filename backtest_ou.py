"""
backtest_ou.py (v2)
Uses real historical DraftKings lines from historical_lines.csv if available.
Falls back to fixed 6.5 line if not.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import norm
import joblib, json, os

JUICE = 1.909  # -110 decimal

# ── Load O/U model ────────────────────────────────────────────────────────────
print("Loading O/U model...")
bundle       = joblib.load('data/model_total.pkl')
features_ou  = bundle['features']
residual_std = bundle['residual_std']
print(f"  Features: {features_ou}")
print(f"  Residual std: {residual_std:.3f}")

# ── Load training data ────────────────────────────────────────────────────────
print("\nLoading training_data.csv...")
df = pd.read_csv('data/training_data.csv')
df = df.sort_values('date').reset_index(drop=True)
df['actual_total'] = df['home_score'] + df['away_score']
features_ou = [f for f in features_ou if f in df.columns]
df = df.dropna(subset=features_ou + ['actual_total'])
print(f"  {len(df)} games after dropna")

# ── Load historical lines if available ───────────────────────────────────────
USE_REAL_LINES = False
if os.path.exists('data/historical_lines.csv'):
    print("\nLoading historical_lines.csv...")
    lines_df = pd.read_csv('data/historical_lines.csv')
    lines_df = lines_df.dropna(subset=['close_line'])
    print(f"  {len(lines_df)} games with real lines")

    # Normalize team names for join
    FULL_TO_ABBR = {
        'Anaheim Ducks':'ANA','Boston Bruins':'BOS','Buffalo Sabres':'BUF',
        'Calgary Flames':'CGY','Carolina Hurricanes':'CAR','Chicago Blackhawks':'CHI',
        'Colorado Avalanche':'COL','Columbus Blue Jackets':'CBJ','Dallas Stars':'DAL',
        'Detroit Red Wings':'DET','Edmonton Oilers':'EDM','Florida Panthers':'FLA',
        'Los Angeles Kings':'LAK','Minnesota Wild':'MIN','Montréal Canadiens':'MTL',
        'Montreal Canadiens':'MTL','Nashville Predators':'NSH','New Jersey Devils':'NJD',
        'New York Rangers':'NYR','New York Islanders':'NYI','Ottawa Senators':'OTT',
        'Philadelphia Flyers':'PHI','Pittsburgh Penguins':'PIT','San Jose Sharks':'SJS',
        'Seattle Kraken':'SEA','St. Louis Blues':'STL','Tampa Bay Lightning':'TBL',
        'Toronto Maple Leafs':'TOR','Utah Mammoth':'UTA','Utah Hockey Club':'UTA',
        'Vancouver Canucks':'VAN','Vegas Golden Knights':'VGK','Washington Capitals':'WSH',
        'Winnipeg Jets':'WPG','Arizona Coyotes':'ARI',
    }
    def to_abbr(t):
        t = str(t).strip()
        return FULL_TO_ABBR.get(t, t)

    df['home_abbr'] = df['home_team'].apply(to_abbr)
    df['away_abbr'] = df['away_team'].apply(to_abbr)

    # Join on date + home + away
    lines_key = lines_df.set_index(['date','home_team','away_team'])['close_line']
    def get_line(row):
        key = (row['date'], row['home_abbr'], row['away_abbr'])
        if key in lines_key.index: return lines_key[key]
        # Try reversed (SBR sometimes swaps)
        key2 = (row['date'], row['away_abbr'], row['home_abbr'])
        if key2 in lines_key.index: return lines_key[key2]
        return np.nan

    df['real_line'] = df.apply(get_line, axis=1)
    matched = df['real_line'].notna().sum()
    pct = matched / len(df) * 100
    print(f"  Matched {matched}/{len(df)} games ({pct:.1f}%) to real lines")

    if matched > 100:
        USE_REAL_LINES = True
        print(f"  ✅ Using real lines for backtest")
        print(f"  Line distribution:\n{df['real_line'].value_counts().sort_index().head(10)}")
    else:
        print(f"  ⚠️  Too few matches — falling back to fixed 6.5")
else:
    print("\n  historical_lines.csv not found — using fixed 6.5 line")
    print("  Run fetch_historical_lines.py to get real lines")

# ── Train/test split ──────────────────────────────────────────────────────────
split    = int(len(df) * 0.80)
train_df = df.iloc[:split].copy()
test_df  = df.iloc[split:].copy()
print(f"\nTrain: {len(train_df)} | Test: {len(test_df)}")
print(f"Test:  {test_df['date'].min()} → {test_df['date'].max()}")

# ── Train fresh model ─────────────────────────────────────────────────────────
X_train = train_df[features_ou]
y_train = train_df['actual_total']
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                   max_depth=3, random_state=42)
model.fit(X_train, y_train)
res_std = float(np.std(y_train - model.predict(X_train)))
print(f"Train residual std: {res_std:.3f}")

# ── Predict ───────────────────────────────────────────────────────────────────
test_df['pred_total'] = model.predict(test_df[features_ou])
print(f"MAE: {np.abs(test_df['pred_total'] - test_df['actual_total']).mean():.3f} goals")

def run_backtest(data, line_series, label):
    data = data.copy()
    data['line']       = line_series
    data = data.dropna(subset=['line'])
    data['over_prob']  = 1 - norm.cdf(data['line'], loc=data['pred_total'], scale=res_std)
    data['pick_over']  = data['over_prob'] >= 0.5
    data['conf']       = np.where(data['pick_over'], data['over_prob'], 1-data['over_prob'])
    data['actual_over']= data['actual_total'] > data['line']
    data['correct']    = data['pick_over'] == data['actual_over']

    print(f"\n{'='*65}")
    print(f"BACKTEST: {label} ({len(data)} games)")
    print(f"{'='*65}")

    over_pct = data['actual_over'].mean()
    print(f"Actual over rate: {over_pct:.1%}")
    print(f"Model picks: {data['pick_over'].sum()} Over / {(~data['pick_over']).sum()} Under")

    w   = int(data['correct'].sum())
    l   = len(data) - w
    acc = data['correct'].mean()
    roi = (w*(JUICE-1)*100 - l*100)/(len(data)*100)*100
    print(f"Overall: {w}-{l} ({acc:.1%}) | ROI: {roi:+.1f}%")
    print(f"Baseline (always over): {over_pct:.1%}")

    print(f"\n{'Band':<10} {'W-L':<10} {'Acc':>6} {'ROI':>8} {'O/U':>8} {'Games':>6}")
    print("-"*55)
    for lo,hi,lbl in [(0.50,0.55,'50-55%'),(0.55,0.60,'55-60%'),
                       (0.60,0.65,'60-65%'),(0.65,0.70,'65-70%'),(0.70,1.01,'70%+')]:
        sub = data[(data['conf']>=lo)&(data['conf']<hi)]
        if len(sub)==0:
            print(f"{lbl:<10} {'—'}")
            continue
        w2  = int(sub['correct'].sum()); l2 = len(sub)-w2
        a2  = sub['correct'].mean()
        r2  = (w2*(JUICE-1)*100 - l2*100)/(len(sub)*100)*100
        o2  = int(sub['pick_over'].sum()); u2 = len(sub)-o2
        flg = ' ✅' if r2>0 else ' ❌'
        print(f"{lbl:<10} {w2}-{l2:<6} {a2:>6.1%} {r2:>+7.1f}%{flg}  O:{o2}/U:{u2}  ({len(sub)})")

    # Monthly
    if 'date' in data.columns:
        print(f"\n{'Month':<10} {'W-L':<10} {'Acc':>6} {'ROI':>8}  (60%+ only)")
        print("-"*45)
        data['month'] = pd.to_datetime(data['date'], errors='coerce').dt.to_period('M')
        hc = data[data['conf'] >= 0.60]
        for mo, grp in hc.groupby('month'):
            w2=int(grp['correct'].sum()); l2=len(grp)-w2
            a2=grp['correct'].mean()
            r2=(w2*(JUICE-1)*100-l2*100)/(len(grp)*100)*100
            print(f"{str(mo):<10} {w2}-{l2:<7} {a2:>6.1%} {r2:>+7.1f}%")

# ── Run backtests ─────────────────────────────────────────────────────────────
if USE_REAL_LINES:
    run_backtest(test_df, test_df['real_line'], "Real DraftKings Lines")
    # Also show fixed 6.5 for comparison
    run_backtest(test_df, pd.Series(6.5, index=test_df.index), "Fixed 6.5 Line (comparison)")
else:
    for line in [6.0, 6.5]:
        run_backtest(test_df, pd.Series(line, index=test_df.index), f"Fixed {line} Line")

# ── Live record ───────────────────────────────────────────────────────────────
if os.path.exists('data/record.json'):
    rec   = json.load(open('data/record.json'))
    ou_at = rec.get('ou_alltime', {})
    tot   = ou_at.get('W',0)+ou_at.get('L',0)
    pct   = ou_at['W']/tot*100 if tot else 0
    print(f"\n{'='*65}")
    print(f"Live O/U record: {ou_at.get('W',0)}-{ou_at.get('L',0)} ({pct:.1f}%) over {tot} picks")
    # By conf band
    by_conf = rec.get('ou_by_conf', {})
    if by_conf:
        print("Live by confidence band:")
        for band, label in [('60','60-65%'),('65','65-70%'),('70','70%+')]:
            b = by_conf.get(band, {})
            t = b.get('W',0)+b.get('L',0)
            p = b['W']/t*100 if t else 0
            print(f"  {label}: {b.get('W',0)}-{b.get('L',0)} ({p:.0f}%)")

print(f"\n{'='*65}")
print("Break-even with -110 juice: 52.4% | Bet threshold: 60%+")