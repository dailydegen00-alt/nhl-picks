"""
fetch_historical_goalies.py (v3)
- 2025: MoneyPuck season summary (works)
- 2023, 2024: NHL API goalie stats (always public)
Builds data/team_goalie_gsax.csv with one row per team per season.
"""
import pandas as pd
import requests

TEAM_NORM = {
    'T.B':'TBL','TB':'TBL','N.J':'NJD','NJ':'NJD',
    'S.J':'SJS','L.A':'LAK','PHX':'ARI',
}
def norm(t):
    return TEAM_NORM.get(str(t).strip().upper(), str(t).strip().upper())

all_rows = []

# ── 2025-26: MoneyPuck (works) ────────────────────────────────────────────────
print("Fetching 2025-26 from MoneyPuck...")
try:
    df = pd.read_csv('https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/goalies.csv')
    df = df[df['situation'] == 'all'].copy()
    df['gsax'] = df['xGoals'] - df['goals']
    df['season_year'] = 2025
    df['team'] = df['team'].apply(norm)
    all_rows.append(df[['name','team','season_year','games_played','gsax']])
    print(f"  2025: {len(df)} goalies")
except Exception as e:
    print(f"  2025 ERROR: {e}")

# ── 2023-24 and 2024-25: NHL API ──────────────────────────────────────────────
for season_id, year in [('20232024', 2023), ('20242025', 2024)]:
    print(f"Fetching {year}-{str(year+1)[2:]} from NHL API...")
    url = (f'https://api.nhle.com/stats/rest/en/goalie/summary'
           f'?limit=-1&cayenneExp=seasonId={season_id}%20and%20gameTypeId=2')
    try:
        data = requests.get(url, timeout=15).json()
        rows = data.get('data', [])
        if not rows:
            print(f"  {year}: no data returned")
            continue

        df = pd.DataFrame(rows)
        print(f"  {year}: {len(df)} rows, cols: {list(df.columns[:10])}")

        # GSAx not directly available from NHL API — use saves above average proxy:
        # We have: saves, shotsAgainst, savePct
        # League avg save pct ~0.906; GSAx ≈ (savePct - 0.906) * shotsAgainst
        LEAGUE_AVG_SV = 0.906
        if 'savePctg' in df.columns and 'shotsAgainst' in df.columns:
            df['gsax'] = (df['savePctg'] - LEAGUE_AVG_SV) * df['shotsAgainst']
        elif 'savePct' in df.columns and 'shotsAgainst' in df.columns:
            df['gsax'] = (df['savePct'] - LEAGUE_AVG_SV) * df['shotsAgainst']
        else:
            sv_col = next((c for c in df.columns if 'save' in c.lower() and 'pct' in c.lower()), None)
            sa_col = next((c for c in df.columns if 'shots' in c.lower() and 'against' in c.lower()), None)
            print(f"  Save% col: {sv_col}, ShotsAgainst col: {sa_col}")
            if sv_col and sa_col:
                df['gsax'] = (df[sv_col] - LEAGUE_AVG_SV) * df[sa_col]
            else:
                print(f"  Could not compute GSAx for {year}")
                continue

        # team column
        team_col = next((c for c in ['teamAbbrevs','teamAbbrev','team'] if c in df.columns), None)
        name_col = next((c for c in ['goalieFullName','fullName','name'] if c in df.columns), None)
        gp_col   = next((c for c in ['gamesPlayed','games_played'] if c in df.columns), None)

        print(f"  Using: team={team_col}, name={name_col}, gp={gp_col}")
        if not team_col or not gp_col:
            print(f"  Missing required columns")
            continue

        df['team']         = df[team_col].apply(norm)
        df['name']         = df[name_col] if name_col else 'Unknown'
        df['games_played'] = df[gp_col]
        df['season_year']  = year
        all_rows.append(df[['name','team','season_year','games_played','gsax']])
        print(f"  {year}: done — {len(df)} goalies")

    except Exception as e:
        print(f"  {year} ERROR: {e}")

# ── Combine and compute team-weighted GSAx ────────────────────────────────────
if not all_rows:
    print("\nNo data. Exiting.")
    exit()

goalies = pd.concat(all_rows, ignore_index=True)
print(f"\nTotal: {len(goalies)} goalie-season rows across {goalies['season_year'].nunique()} seasons")

def wavg(grp):
    gp = grp['games_played'].clip(lower=0).sum()
    return (grp['gsax'] * grp['games_played'].clip(lower=0)).sum() / gp if gp > 0 else 0.0

team_gsax = (
    goalies.groupby(['team','season_year'])
    .apply(wavg, include_groups=False)
    .reset_index()
)
team_gsax.columns = ['team','season_year','team_goalie_gsax']

print(f"\nTop 10 teams by GSAx:")
print(team_gsax.sort_values('team_goalie_gsax', ascending=False).head(10).to_string(index=False))
print(f"\nSeasons covered: {sorted(team_gsax['season_year'].unique())}")

team_gsax.to_csv('data/team_goalie_gsax.csv', index=False)
print("\nSaved data/team_goalie_gsax.csv")
print("Next: python add_goalie_training.py")