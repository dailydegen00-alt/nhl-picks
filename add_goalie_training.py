"""
add_goalie_training.py (v4)
Joins team-season GSAx to training_data.csv.
Handles full team names in training_data (e.g. "Seattle Kraken" → "SEA").
"""
import pandas as pd
import numpy as np

# Full name → abbreviation
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
    'Winnipeg Jets':'WPG','Arizona Coyotes':'ARI','Phoenix Coyotes':'ARI',
}

TEAM_NORM = {
    'T.B':'TBL','TB':'TBL','N.J':'NJD','NJ':'NJD',
    'S.J':'SJS','L.A':'LAK','PHX':'ARI',
}

def norm(t):
    t = str(t).strip()
    # Try full name first
    if t in FULL_TO_ABBR:
        return FULL_TO_ABBR[t]
    t_up = t.upper()
    return TEAM_NORM.get(t_up, t_up)

def date_to_season_year(date_str):
    d = pd.to_datetime(date_str)
    return d.year if d.month >= 8 else d.year - 1

# ── Load GSAx lookup ──────────────────────────────────────────────────────────
print("Loading team_goalie_gsax.csv...")
try:
    gsax_df = pd.read_csv('data/team_goalie_gsax.csv')
    gsax_df['team'] = gsax_df['team'].apply(norm)
    seasons = sorted(gsax_df['season_year'].unique())
    print(f"  {len(gsax_df)} rows | seasons: {seasons}")
    print(f"  Sample teams: {sorted(gsax_df['team'].unique())[:8]}")
except FileNotFoundError:
    print("  ERROR: run fetch_historical_goalies.py first")
    exit()

gsax_map = {
    (row['team'], int(row['season_year'])): row['team_goalie_gsax']
    for _, row in gsax_df.iterrows()
}

# ── Load training data ────────────────────────────────────────────────────────
print("\nLoading training_data.csv...")
train = pd.read_csv('data/training_data.csv')
print(f"  {len(train)} games | {train['date'].min()} → {train['date'].max()}")

# Remove old goalie columns if they exist (so we start clean)
for col in ['home_goalie_gsax','away_goalie_gsax','gsax_diff']:
    if col in train.columns:
        train.drop(columns=[col], inplace=True)

# Show sample team names from training data
sample_teams = set(train['home_team'].unique()[:5]) | set(train['away_team'].unique()[:5])
print(f"  Sample team names in training data: {list(sample_teams)[:6]}")

# ── Join ──────────────────────────────────────────────────────────────────────
print("\nJoining goalie GSAx...")
h_list, a_list = [], []
matched = missed = 0
miss_examples = []

for _, row in train.iterrows():
    yr   = date_to_season_year(row['date'])
    home = norm(row['home_team'])
    away = norm(row['away_team'])

    h_val = gsax_map.get((home, yr), np.nan)
    a_val = gsax_map.get((away, yr), np.nan)
    h_list.append(h_val)
    a_list.append(a_val)

    if not np.isnan(h_val) and not np.isnan(a_val):
        matched += 1
    else:
        missed += 1
        if len(miss_examples) < 3:
            miss_examples.append(f"{row['home_team']} ({home}) vs {row['away_team']} ({away}) yr={yr}")

train['home_goalie_gsax'] = h_list
train['away_goalie_gsax'] = a_list
train['gsax_diff']        = train['home_goalie_gsax'] - train['away_goalie_gsax']

pct = 100 * matched / len(train)
print(f"  Matched: {matched}/{len(train)} ({pct:.1f}%)")
if miss_examples:
    print(f"  Unmatched examples (showing norm'd abbr → lookup key):")
    for e in miss_examples:
        print(f"    {e}")

# Fill nulls with 0
for col in ['home_goalie_gsax','away_goalie_gsax','gsax_diff']:
    n = train[col].isna().sum()
    train[col] = train[col].fillna(0.0)
    if n: print(f"  Filled {n} nulls in {col} with 0")

print("\nGSAx distribution:")
print(train[['home_goalie_gsax','away_goalie_gsax','gsax_diff']].describe().round(2))

train.to_csv('data/training_data.csv', index=False)
print(f"\nSaved training_data.csv — {len(train.columns)} features")
print("Next: python model.py")