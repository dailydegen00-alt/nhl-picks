import pandas as pd

name_map = {
    'Anaheim':'Anaheim Ducks','Boston':'Boston Bruins','Buffalo':'Buffalo Sabres',
    'Calgary':'Calgary Flames','Carolina':'Carolina Hurricanes','Chicago':'Chicago Blackhawks',
    'Colorado':'Colorado Avalanche','Columbus':'Columbus Blue Jackets','Dallas':'Dallas Stars',
    'Detroit':'Detroit Red Wings','Edmonton':'Edmonton Oilers','Florida':'Florida Panthers',
    'Los Angeles':'Los Angeles Kings','Minnesota':'Minnesota Wild','Montréal':'Montréal Canadiens',
    'Nashville':'Nashville Predators','New Jersey':'New Jersey Devils','New York':'New York Rangers',
    'Ottawa':'Ottawa Senators','Philadelphia':'Philadelphia Flyers','Pittsburgh':'Pittsburgh Penguins',
    'San Jose':'San Jose Sharks','Seattle':'Seattle Kraken','St. Louis':'St. Louis Blues',
    'Tampa Bay':'Tampa Bay Lightning','Toronto':'Toronto Maple Leafs','Utah':'Utah Mammoth',
    'Vancouver':'Vancouver Canucks','Vegas':'Vegas Golden Knights','Washington':'Washington Capitals',
    'Winnipeg':'Winnipeg Jets'
}

abbr_to_full = {
    'ANA':'Anaheim Ducks','BOS':'Boston Bruins','BUF':'Buffalo Sabres',
    'CGY':'Calgary Flames','CAR':'Carolina Hurricanes','CHI':'Chicago Blackhawks',
    'COL':'Colorado Avalanche','CBJ':'Columbus Blue Jackets','DAL':'Dallas Stars',
    'DET':'Detroit Red Wings','EDM':'Edmonton Oilers','FLA':'Florida Panthers',
    'LAK':'Los Angeles Kings','MIN':'Minnesota Wild','MTL':'Montréal Canadiens',
    'NSH':'Nashville Predators','NJD':'New Jersey Devils','NYR':'New York Rangers',
    'NYI':'New York Islanders','OTT':'Ottawa Senators','PHI':'Philadelphia Flyers',
    'PIT':'Pittsburgh Penguins','SJS':'San Jose Sharks','SEA':'Seattle Kraken',
    'STL':'St. Louis Blues','TBL':'Tampa Bay Lightning','TOR':'Toronto Maple Leafs',
    'UTA':'Utah Mammoth','VAN':'Vancouver Canucks','VGK':'Vegas Golden Knights',
    'WSH':'Washington Capitals','WPG':'Winnipeg Jets',
    'T.B':'Tampa Bay Lightning','S.J':'San Jose Sharks','N.J':'New Jersey Devils',
    'L.A':'Los Angeles Kings','TB':'Tampa Bay Lightning','SJ':'San Jose Sharks',
    'NJ':'New Jersey Devils','LA':'Los Angeles Kings','ARI':'Utah Mammoth',
}

def load_mp(path):
    df = pd.read_csv(path)

    # ── 5on5 stats ────────────────────────────
    df5 = df[df['situation'] == '5on5'].copy()
    df5['full_name'] = df5['name'].map(abbr_to_full)
    df5 = df5.dropna(subset=['full_name'])

    stats = df5[['full_name','xGoalsPercentage','corsiPercentage',
                 'fenwickPercentage','scoreVenueAdjustedxGoalsFor',
                 'scoreVenueAdjustedxGoalsAgainst',
                 'highDangerShotsFor','highDangerShotsAgainst',
                 'games_played']].copy()
    stats.columns = ['team','xg_pct','corsi_pct','fenwick_pct',
                     'sv_xgf','sv_xga','hdcf','hdca','gp']

    for col in ['sv_xgf','sv_xga','hdcf','hdca']:
        stats[col] = stats[col] / stats['gp'].replace(0,1)

    stats['sv_xg_pct'] = stats['sv_xgf'] / (stats['sv_xgf'] + stats['sv_xga']).replace(0,1)
    stats['hdc_pct']   = stats['hdcf']   / (stats['hdcf']   + stats['hdca']).replace(0,1)

    # ── PDO from 'all' ────────────────────────
    dfa = df[df['situation'] == 'all'].copy()
    dfa['full_name'] = dfa['name'].map(abbr_to_full)
    dfa = dfa.dropna(subset=['full_name'])

    if 'goalsFor' in dfa.columns and 'shotsOnGoalFor' in dfa.columns:
        dfa['sh_pct'] = dfa['goalsFor']     / dfa['shotsOnGoalFor'].replace(0,1)
        dfa['sv_pct'] = 1 - (dfa['goalsAgainst'] / dfa['shotsOnGoalAgainst'].replace(0,1))
        dfa['pdo']    = dfa['sh_pct'] + dfa['sv_pct']
        pdo = dfa[['full_name','pdo']].rename(columns={'full_name':'team'})
        stats = stats.merge(pdo, on='team', how='left')
    else:
        stats['pdo'] = 1.0

    # ── Power play stats (5on4) ───────────────
    pp = df[df['situation'] == '5on4'].copy()
    pp['full_name'] = pp['name'].map(abbr_to_full)
    pp = pp.dropna(subset=['full_name'])

    if 'xGoalsFor' in pp.columns and 'iceTime' in pp.columns:
        pp['pp_xgf_per60']   = pp['xGoalsFor']   / (pp['iceTime'] / 60).replace(0,1)
        pp['pp_min_per_game'] = (pp['iceTime'] / 60) / pp['games_played'].replace(0,1)
        pp_stats = pp[['full_name','pp_xgf_per60','pp_min_per_game']].rename(
            columns={'full_name':'team'})
        stats = stats.merge(pp_stats, on='team', how='left')
    else:
        stats['pp_xgf_per60']   = 0.12
        stats['pp_min_per_game'] = 3.5

    # ── Penalty kill stats (4on5) ─────────────
    pk = df[df['situation'] == '4on5'].copy()
    pk['full_name'] = pk['name'].map(abbr_to_full)
    pk = pk.dropna(subset=['full_name'])

    if 'xGoalsAgainst' in pk.columns and 'iceTime' in pk.columns:
        pk['pk_xga_per60']   = pk['xGoalsAgainst'] / (pk['iceTime'] / 60).replace(0,1)
        pk['pk_min_per_game'] = (pk['iceTime'] / 60) / pk['games_played'].replace(0,1)
        pk_stats = pk[['full_name','pk_xga_per60','pk_min_per_game']].rename(
            columns={'full_name':'team'})
        stats = stats.merge(pk_stats, on='team', how='left')
    else:
        stats['pk_xga_per60']   = 0.12
        stats['pk_min_per_game'] = 3.5

    stats = stats.drop(columns=['gp'])
    return stats.set_index('team')

def load_games(path):
    df = pd.read_csv(path)
    df['home_team'] = df['home_team'].map(name_map)
    df['away_team'] = df['away_team'].map(name_map)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True).dropna(
        subset=['home_team','away_team'])

def add_features(games, team_stats):
    games = games.copy()

    # ── Rolling form (per game log) ────────────────────────────────────────────
    home_log = games[['date','home_team','home_win','home_score','away_score']].rename(
        columns={'home_team':'team','home_win':'win',
                 'home_score':'gf','away_score':'ga'})
    away_log = games[['date','away_team','home_win','away_score','home_score']].copy()
    away_log['win'] = 1 - away_log['home_win']
    away_log = away_log.rename(
        columns={'away_team':'team','away_score':'gf','home_score':'ga'}
    )[['date','team','win','gf','ga']]

    log = pd.concat([home_log, away_log]).sort_values('date').reset_index(drop=True)
    log['goal_diff'] = log['gf'] - log['ga']

    # Last 10 game rolling stats
    log['form_l10']     = log.groupby('team')['win'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    log['gpg_l10']      = log.groupby('team')['gf'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    log['gapg_l10']     = log.groupby('team')['ga'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    log['gdiff_l10']    = log.groupby('team')['goal_diff'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean())

    # Last 5 game rolling stats (shorter window = more reactive)
    log['form_l5']      = log.groupby('team')['win'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).mean())
    log['gpg_l5']       = log.groupby('team')['gf'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).mean())
    log['gdiff_l5']     = log.groupby('team')['goal_diff'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).mean())

    # Merge rolling stats back
    roll_cols = ['form_l10','gpg_l10','gapg_l10','gdiff_l10','form_l5','gpg_l5','gdiff_l5']

    for col in roll_cols:
        agg = log.groupby(['date','team'])[col].mean().reset_index()

        games = games.merge(
            agg.rename(columns={'team':'home_team', col:f'home_{col}'}),
            on=['date','home_team'], how='left')
        games = games.merge(
            agg.rename(columns={'team':'away_team', col:f'away_{col}'}),
            on=['date','away_team'], how='left')

    # Keep backward compat column names
    games['home_form'] = games['home_form_l10']
    games['away_form'] = games['away_form_l10']
    games['home_gpg']  = games['home_gpg_l10']
    games['away_gpg']  = games['away_gpg_l10']

    # Rolling differentials
    games['form_diff']      = games['home_form_l10']  - games['away_form_l10']
    games['gpg_diff']       = games['home_gpg_l10']   - games['away_gpg_l10']
    games['gapg_diff']      = games['home_gapg_l10']  - games['away_gapg_l10']
    games['gdiff_l10_diff'] = games['home_gdiff_l10'] - games['away_gdiff_l10']
    games['form_l5_diff']   = games['home_form_l5']   - games['away_form_l5']
    games['gdiff_l5_diff']  = games['home_gdiff_l5']  - games['away_gdiff_l5']

    # ── Season stats from MoneyPuck ────────────────────────────────────────────
    ts = team_stats.reset_index()
    ts.columns = ['home_team'] + [f'home_{c}' for c in team_stats.columns]
    games = games.merge(ts, on='home_team', how='left')

    ts2 = team_stats.reset_index()
    ts2.columns = ['away_team'] + [f'away_{c}' for c in team_stats.columns]
    games = games.merge(ts2, on='away_team', how='left')

    # Standard differentials
    for stat in ['xg_pct','corsi_pct','fenwick_pct','sv_xg_pct','hdc_pct','pdo']:
        h, a = f'home_{stat}', f'away_{stat}'
        if h in games.columns and a in games.columns:
            games[f'{stat}_diff'] = games[h] - games[a]

    # ── Special teams ──────────────────────────────────────────────────────────
    if 'home_pp_xgf_per60' in games.columns:
        games['home_pp_threat'] = (
            games['home_pp_xgf_per60'] * games['away_pk_min_per_game'])
        games['away_pp_threat'] = (
            games['away_pp_xgf_per60'] * games['home_pk_min_per_game'])
        games['pp_threat_diff'] = games['home_pp_threat'] - games['away_pp_threat']

        games['home_pk_exposure'] = (
            games['away_pp_xgf_per60'] * games['home_pk_min_per_game'])
        games['away_pk_exposure'] = (
            games['home_pp_xgf_per60'] * games['away_pk_min_per_game'])
        games['pk_exposure_diff'] = games['home_pk_exposure'] - games['away_pk_exposure']

    return games.dropna(subset=['home_xg_pct','away_xg_pct','home_form'])


print("Loading seasons...")

df_2324 = add_features(load_games('data/games_2324.csv'), load_mp('data/mp_2023.csv'))
df_2324['season'] = '2023-24'
print(f"  2023-24: {len(df_2324)} games")

df_2425 = add_features(load_games('data/games_2425.csv'), load_mp('data/mp_2024.csv'))
df_2425['season'] = '2024-25'
print(f"  2024-25: {len(df_2425)} games")

df_curr = add_features(load_games('data/games.csv'), load_mp('data/moneypuck.csv'))
df_curr['season'] = '2025-26'
print(f"  2025-26: {len(df_curr)} games")

combined = pd.concat([df_2324, df_2425, df_curr], ignore_index=True)
combined.to_csv('data/training_data.csv', index=False)

print(f"\nTotal: {len(combined)} games")
print(f"Home win rate: {combined['home_win'].mean():.1%}")

new_cols = [c for c in combined.columns if 'l5' in c or 'l10' in c or 'gapg' in c]
print(f"\nNew rolling features: {new_cols}")

st_cols = [c for c in combined.columns if 'pp' in c or 'pk' in c or 'threat' in c]
print(f"Special teams features: {st_cols}")