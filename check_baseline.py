import pandas as pd

df = pd.read_csv('data/training_data.csv')
df = df.sort_values('date').reset_index(drop=True)

total = len(df)
home_wins = df['home_win'].sum()
home_pct = home_wins / total

print(f"Total games: {total}")
print(f"Home wins: {home_wins} ({home_pct:.1%})")
print(f"Away wins: {total - home_wins} ({1-home_pct:.1%})")
print()

# Check by season
for season, grp in df.groupby('season'):
    hw = grp['home_win'].mean()
    print(f"{season}: home wins {hw:.1%}")

print()
print("So just picking home every game = 55.6%")
print("Just picking away every game = 44.4%")
print()

# What if we use point differential as a simple favourite predictor?
# Teams with more points = favourite
# Check how often the team with better xg_pct_diff wins
if 'xg_pct_diff' in df.columns:
    df2 = df.dropna(subset=['xg_pct_diff','home_win'])
    # Pick home if xg_pct_diff > 0, else pick away
    df2['xg_pick_correct'] = ((df2['xg_pct_diff'] > 0) == (df2['home_win'] == 1))
    print(f"Just picking better xG team: {df2['xg_pick_correct'].mean():.1%}")

if 'sv_xg_pct_diff' in df.columns:
    df3 = df.dropna(subset=['sv_xg_pct_diff','home_win'])
    df3['sv_pick_correct'] = ((df3['sv_xg_pct_diff'] > 0) == (df3['home_win'] == 1))
    print(f"Just picking better sv_xG team: {df3['sv_pick_correct'].mean():.1%}")

if 'pdo_diff' in df.columns:
    df4 = df.dropna(subset=['pdo_diff','home_win'])
    df4['pdo_pick_correct'] = ((df4['pdo_diff'] > 0) == (df4['home_win'] == 1))
    print(f"Just picking better PDO team: {df4['pdo_pick_correct'].mean():.1%}")
