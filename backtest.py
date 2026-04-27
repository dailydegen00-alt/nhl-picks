"""
backtest.py (v7)
- Train: training_data seasons 2023 + 2024 (2023-24, 2024-25)
- Test:  training_data season 2025 (2025-26) — already has correct features
- Odds:  historical_ml_odds.csv (Oct 2025–Apr 2026) — matches test set dates
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib, json, os

JUICE = 1.909  # -110 fallback

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
    'Utah Mammoth':'UTA','Utah Hockey Club':'UTA','Vancouver Canucks':'VAN',
    'Vegas Golden Knights':'VGK','Washington Capitals':'WSH','Winnipeg Jets':'WPG',
    'Arizona Coyotes':'ARI',
}
def to_abbr(t):
    t = str(t).strip()
    return FULL_TO_ABBR.get(t, t)

def season_year(d):
    d = pd.to_datetime(d)
    return d.year if d.month >= 8 else d.year - 1

# ── Load model ────────────────────────────────────────────────────────────────
saved    = joblib.load('data/model.pkl')
FEATURES = saved['features']
print(f"Features: {FEATURES}\n")

# ── Load training data ────────────────────────────────────────────────────────
print("Loading training_data.csv...")
df = pd.read_csv('data/training_data.csv').sort_values('date').reset_index(drop=True)
df['sy'] = df['date'].apply(season_year)
FEATURES = [f for f in FEATURES if f in df.columns]
df = df.dropna(subset=FEATURES + ['home_win'])

seasons = df['sy'].value_counts().sort_index().to_dict()
print(f"  {len(df)} games | seasons: {seasons}")

train_df = df[df['sy'].isin([2023, 2024])].copy()
test_df  = df[df['sy'] == 2025].copy()

print(f"  Train: {len(train_df)} games ({train_df['date'].min()} → {train_df['date'].max()})")
print(f"  Test:  {len(test_df)} games ({test_df['date'].min()} → {test_df['date'].max()})")

if len(test_df) == 0:
    print("  No 2025-26 games found — using last 20% of training data")
    split = int(len(df)*0.8)
    train_df, test_df = df.iloc[:split].copy(), df.iloc[split:].copy()

# ── Load real odds ────────────────────────────────────────────────────────────
odds_map = {}
if os.path.exists('data/historical_ml_odds.csv'):
    odds_df = pd.read_csv('data/historical_ml_odds.csv')
    for _, r in odds_df.iterrows():
        odds_map[(r['date'], r['home_team'], r['away_team'])] = {
            'home_dec': float(r['home_odds_dec']),
            'away_dec': float(r['away_odds_dec']),
            'home_imp': float(r['home_implied']),
            'away_imp': float(r['away_implied']),
        }
    print(f"\nLoaded {len(odds_map)} real odds entries")

# Add abbreviation columns to test_df for odds matching
test_df['home_abbr'] = test_df['home_team'].apply(to_abbr)
test_df['away_abbr'] = test_df['away_team'].apply(to_abbr)

matched = sum(
    1 for _, r in test_df.iterrows()
    if (r['date'], r['home_abbr'], r['away_abbr']) in odds_map
)
print(f"Real odds matched: {matched}/{len(test_df)} ({matched/len(test_df)*100:.1f}%)")

# Sample unmatched to debug if needed
if matched == 0 and len(test_df) > 0:
    r0 = test_df.iloc[0]
    key0 = (r0['date'], r0['home_abbr'], r0['away_abbr'])
    print(f"  First test game key: {key0}")
    # Show a few odds keys for comparison
    sample_keys = list(odds_map.keys())[:3]
    print(f"  Sample odds keys: {sample_keys}")

# ── Train + predict ───────────────────────────────────────────────────────────
model = Pipeline([('s', StandardScaler()), ('m', LogisticRegression(max_iter=1000))])
model.fit(train_df[FEATURES], train_df['home_win'])

probs = model.predict_proba(test_df[FEATURES])[:,1]
test_df = test_df.copy()
test_df['prob']      = probs
test_df['conf']      = np.where(probs>=0.5, probs, 1-probs)
test_df['pick_home'] = probs >= 0.5
test_df['correct']   = (
    (test_df['pick_home'] & (test_df['home_win']==1)) |
    (~test_df['pick_home'] & (test_df['home_win']==0))
)

def sim(subset):
    if len(subset)==0: return 0,0,0.0,0.0
    w=int(subset['correct'].sum()); l=len(subset)-w
    acc=subset['correct'].mean(); profit=0
    for _,r in subset.iterrows():
        o=odds_map.get((r['date'], r.get('home_abbr',''), r.get('away_abbr','')))
        dec=(o['home_dec'] if r['pick_home'] else o['away_dec']) if o else JUICE
        profit+=(dec-1)*100 if r['correct'] else -100
    return w,l,acc,profit/(len(subset)*100)*100

print(f"\n{'='*60}")
print(f"ML BACKTEST — 2025-26 True Out-of-Sample")
odds_note = f"real odds ({matched} games matched)" if matched>100 else "fixed -110 juice (run parse_historical_ml.py first)"
print(f"Odds: {odds_note}")
print(f"{'='*60}")

w,l,acc,roi=sim(test_df)
print(f"\nAll: {w}-{l} ({acc:.1%}) | ROI: {roi:+.1f}% | {len(test_df)} games")
print(f"Baseline (always home): {test_df['home_win'].mean():.1%}")
print(f"Break-even: 52.4%")

print(f"\n{'Band':<10} {'W-L':<10} {'Acc':>6} {'ROI':>8} {'Games':>7}")
print("-"*46)
for lo,hi,lbl in [(0.50,0.55,'50-55%'),(0.55,0.60,'55-60%'),
                   (0.60,0.65,'60-65%'),(0.65,0.70,'65-70%'),(0.70,1.01,'70%+')]:
    sub=test_df[(test_df['conf']>=lo)&(test_df['conf']<hi)]
    if len(sub)==0: print(f"{lbl:<10} {'—'}"); continue
    w2,l2,a2,r2=sim(sub)
    print(f"{lbl:<10} {w2}-{l2:<6} {a2:>6.1%} {r2:>+7.1f}%{'✅' if r2>0 else '❌'}  ({len(sub)})")

print(f"\n{'Month':<10} {'W-L':<10} {'Acc':>6} {'ROI':>8}  (60%+ conf)")
print("-"*44)
test_df['month']=pd.to_datetime(test_df['date']).dt.to_period('M')
for mo,grp in test_df[test_df['conf']>=0.60].groupby('month'):
    w2,l2,a2,r2=sim(grp)
    print(f"{str(mo):<10} {w2}-{l2:<7} {a2:>6.1%} {r2:>+7.1f}%")

# CLV check
if matched > 50:
    cw=ct=0
    for _,r in test_df.iterrows():
        o=odds_map.get((r['date'],r.get('home_abbr',''),r.get('away_abbr','')))
        if not o: continue
        book=o['home_imp']/100 if r['pick_home'] else o['away_imp']/100
        if r['conf']>book: cw+=1
        ct+=1
    if ct:
        print(f"\nCLV: model beats book on {cw}/{ct} ({cw/ct*100:.1f}%) picks")
        print(f"     >50% = model finds consistent edge vs closing line")

if os.path.exists('data/record.json'):
    rec=json.load(open('data/record.json'))
    at=rec.get('alltime',{})
    tot=at.get('W',0)+at.get('L',0)
    print(f"\nLive record: {at.get('W',0)}-{at.get('L',0)} ({at['W']/tot*100:.1f}%) over {tot} picks")
print(f"{'='*60}")