"""
fetch_playoff_games.py (v3)
Fixes: defines load_mp locally instead of importing build_historical_features
(which would re-run the entire script and wipe training_data.csv).
"""
import requests
import pandas as pd
import numpy as np

# ── load_mp defined locally (avoids importing build_historical_features) ──────
def load_mp(path):
    """Load MoneyPuck team stats, 5on5 situation, indexed by full team name."""
    df = pd.read_csv(path)
    if 'situation' in df.columns:
        df = df[df['situation'] == '5on5']
    if 'team' in df.columns:
        df = df.set_index('team')
    return df

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
ABBR_TO_FULL = {v: k for k, v in FULL_TO_ABBR.items()}

def date_to_season_year(date_str):
    d = pd.to_datetime(date_str)
    return d.year if d.month >= 8 else d.year - 1

def fetch_playoff_games(start_year):
    games = []
    seen  = set()
    for month in [4, 5, 6, 7]:
        for day in range(1, 32):
            try:
                date = f'{start_year+1}-{month:02d}-{day:02d}'
                r = requests.get(
                    f'https://api-web.nhle.com/v1/schedule/{date}', timeout=10
                ).json()
                for week in r.get('gameWeek', []):
                    if week['date'] != date:
                        continue
                    for g in week.get('games', []):
                        if g.get('gameType') != 3:
                            continue
                        if g.get('gameState') not in ('OFF', 'FINAL'):
                            continue
                        key = (date, g['homeTeam']['abbrev'], g['awayTeam']['abbrev'])
                        if key in seen:
                            continue
                        seen.add(key)
                        games.append({
                            'date':            date,
                            'home_team':       g['homeTeam']['abbrev'],
                            'away_team':       g['awayTeam']['abbrev'],
                            'home_score':      g['homeTeam'].get('score', 0),
                            'away_score':      g['awayTeam'].get('score', 0),
                            'home_win':        int(g['homeTeam'].get('score',0) > g['awayTeam'].get('score',0)),
                            'is_playoffs':     1,
                            'season_phase':    5,
                            'season_progress': 1.1,
                        })
            except:
                pass
    return games

# ── Load existing training data ───────────────────────────────────────────────
print("Loading training_data.csv...")
train = pd.read_csv('data/training_data.csv')
po = int(train['is_playoffs'].sum()) if 'is_playoffs' in train.columns else 0
print(f"  {len(train)} games | {train['date'].min()} → {train['date'].max()}")
print(f"  Columns: {len(train.columns)} | Playoff games already in data: {po}")
print(f"  New features present: { {c: c in train.columns for c in ['home_goalie_gsax','season_phase','is_playoffs']} }")

existing = set(zip(train['date'], train['home_team'], train['away_team']))

# ── MoneyPuck stats ───────────────────────────────────────────────────────────
print("\nLoading MoneyPuck stats...")
mp_by_season = {
    2023: load_mp('data/mp_2023.csv'),
    2024: load_mp('data/mp_2024.csv'),
    2025: load_mp('data/moneypuck.csv'),
}

# ── Fetch playoff games ───────────────────────────────────────────────────────
all_new = []
for start_year, label in [(2023,'2023-24'), (2024,'2024-25')]:
    print(f"\nFetching {label} playoffs...")
    games = fetch_playoff_games(start_year)
    new = []
    for g in games:
        hf = ABBR_TO_FULL.get(g['home_team'], g['home_team'])
        af = ABBR_TO_FULL.get(g['away_team'], g['away_team'])
        if (g['date'], hf, af) not in existing and (g['date'], g['home_team'], g['away_team']) not in existing:
            new.append(g)
    print(f"  {len(games)} found, {len(new)} new")
    all_new.extend(new)

if not all_new:
    print("\nNo new playoff games to add.")
else:
    print(f"\nBuilding features for {len(all_new)} games...")
    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    col_medians  = train[numeric_cols].median()

    new_rows = []
    for g in all_new:
        sy  = date_to_season_year(g['date'])
        mp  = mp_by_season.get(sy)
        ha  = g['home_team']
        aa  = g['away_team']
        hf  = ABBR_TO_FULL.get(ha, ha)
        af  = ABBR_TO_FULL.get(aa, aa)

        def gs(abbr, col):
            try:
                full = ABBR_TO_FULL.get(abbr, abbr)
                if full in mp.index: return float(mp.loc[full, col])
                if abbr in mp.index: return float(mp.loc[abbr, col])
            except: pass
            return np.nan

        row = {
            'date':            g['date'],
            'home_team':       hf,
            'away_team':       af,
            'home_score':      g['home_score'],
            'away_score':      g['away_score'],
            'home_win':        g['home_win'],
            'is_playoffs':     1,
            'season_phase':    5,
            'season_progress': 1.1,
        }

        for col in train.columns:
            if col in row:
                continue
            if col.startswith('home_'):
                row[col] = gs(ha, col[5:])
            elif col.startswith('away_'):
                row[col] = gs(aa, col[5:])
            elif col.endswith('_diff'):
                base = col[:-5]
                h = gs(ha, base); a = gs(aa, base)
                row[col] = (h-a) if not (np.isnan(h) or np.isnan(a)) else np.nan
            else:
                row[col] = np.nan

        new_rows.append(row)

    new_df = pd.DataFrame(new_rows).reindex(columns=train.columns, fill_value=np.nan)

    # Fill NaN — numeric columns only
    for col in new_df.columns:
        if new_df[col].isna().any() and col in col_medians.index:
            new_df[col] = new_df[col].fillna(col_medians[col])

    combined = pd.concat([train, new_df], ignore_index=True).sort_values('date').reset_index(drop=True)
    combined.to_csv('data/training_data.csv', index=False)

    po_final = int(combined['is_playoffs'].sum()) if 'is_playoffs' in combined.columns else 0
    print(f"\nSaved — {len(train)} existing + {len(new_df)} playoff = {len(combined)} total")
    print(f"Playoff games (is_playoffs=1): {po_final}")
    print(f"Columns: {len(combined.columns)}")

print("\nNext: python model.py")