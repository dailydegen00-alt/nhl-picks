"""
add_season_phase.py (v2)
Fixes playoff detection — uses April 19+ cutoff not month >= 5.
"""
import pandas as pd
import numpy as np

def date_to_season_year(date_str):
    d = pd.to_datetime(date_str)
    return d.year if d.month >= 8 else d.year - 1

def is_playoff(date_str):
    """Playoffs start ~April 19 each year. Regular season ends ~April 18."""
    d = pd.to_datetime(date_str)
    # Months June/July are always playoffs (conference finals / SCF)
    # April 19+ is roughly when playoffs begin
    if d.month in (5, 6, 7):
        return True
    if d.month == 4 and d.day >= 19:
        return True
    return False

print("Loading training_data.csv...")
train = pd.read_csv('data/training_data.csv')

# Drop old phase columns if they exist
for col in ['season_phase','season_progress','is_playoffs','season_year']:
    if col in train.columns:
        train.drop(columns=[col], inplace=True)

print(f"  {len(train)} games | {train['date'].min()} → {train['date'].max()}")

train['season_year']  = train['date'].apply(date_to_season_year)
train['date_dt']      = pd.to_datetime(train['date'])
train['is_playoffs']  = train['date'].apply(is_playoff).astype(int)

# Game-day rank within each season (regular season only for ranking)
reg = train[train['is_playoffs'] == 0].copy()
season_game_counts = {}
for sy in reg['season_year'].unique():
    dates = sorted(reg[reg['season_year'] == sy]['date'].unique())
    season_game_counts[sy] = {d: i+1 for i, d in enumerate(dates)}

# Max regular season game days per season
season_max = {}
for sy, dc in season_game_counts.items():
    season_max[sy] = max(dc.values()) if dc else 130

def get_game_num(row):
    if row['is_playoffs']:
        return 999
    sy = row['season_year']
    return season_game_counts.get(sy, {}).get(row['date'], 41)

train['game_num_proxy'] = train.apply(get_game_num, axis=1)

def assign_phase(row):
    if row['is_playoffs']:
        return 5
    sy  = row['season_year']
    gn  = row['game_num_proxy']
    mx  = season_max.get(sy, 130)
    pct = gn / mx
    if pct < 0.15:   return 1   # Early (Oct-Nov)
    elif pct < 0.45: return 2   # Mid (Nov-Jan)
    elif pct < 0.70: return 3   # Late (Feb-Mar)
    else:            return 4   # Crunch (Apr regular season)

train['season_phase']    = train.apply(assign_phase, axis=1)
train['season_progress'] = train.apply(
    lambda r: 1.1 if r['is_playoffs']
    else (r['game_num_proxy'] / season_max.get(r['season_year'], 130)),
    axis=1
).clip(0, 1.2)

# ── Sanity check ──────────────────────────────────────────────────────────────
print("\nPhase distribution:")
labels = {1:'Early',2:'Mid',3:'Late',4:'Crunch',5:'Playoffs'}
for ph, cnt in train['season_phase'].value_counts().sort_index().items():
    sample = train[train['season_phase']==ph]['date'].iloc[0]
    print(f"  Phase {ph} ({labels[ph]:<9}): {cnt:4d} games  (first: {sample})")

# Clean up helper cols
train.drop(columns=['date_dt','game_num_proxy','season_year'], inplace=True, errors='ignore')

train.to_csv('data/training_data.csv', index=False)
print(f"\nSaved — {len(train.columns)} features")
print("Next: python fetch_playoff_games.py")