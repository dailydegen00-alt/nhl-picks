import requests
import pandas as pd

def get_all_injuries():
    """Pull all NHL injuries from ESPN — returns dict keyed by team abbrev."""
    url = 'https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/injuries'
    data = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15).json()

    # ESPN abbrev → NHL abbrev fixes
    espn_fix = {
        'WSH': 'WSH', 'TB': 'TBL', 'SJ': 'SJS', 'NJ': 'NJD',
        'LA': 'LAK', 'CLB': 'CBJ', 'ARI': 'UTA', 'VGK': 'VGK',
    }

    injuries_by_team = {}
    for team_block in data.get('injuries', []):
        abbr = team_block.get('displayName', '')
        # Get abbrev from first injury's team field
        inj_list = team_block.get('injuries', [])
        if not inj_list:
            continue
        team_abbr = inj_list[0].get('athlete', {}).get('team', {}).get('abbreviation', '')
        team_abbr = espn_fix.get(team_abbr, team_abbr)
        if not team_abbr:
            continue

        players = []
        for inj in inj_list:
            athlete  = inj.get('athlete', {})
            name     = athlete.get('displayName', '')
            status   = inj.get('status', '')
            inj_type = inj.get('details', {}).get('type', '')
            ret_date = inj.get('details', {}).get('returnDate', '')
            comment  = inj.get('shortComment', '')

            players.append({
                'name':     name,
                'status':   status,
                'type':     inj_type,
                'return':   ret_date[:10] if ret_date else '',
                'comment':  comment,
            })

        injuries_by_team[team_abbr] = players

    return injuries_by_team

def get_injury_impact(team_abbr, injuries_by_team, skaters_df):
    """
    Cross-reference injured players with MoneyPuck gameScore data.
    Returns (probability_adjustment, list of impactful missing players)
    """
    injured = injuries_by_team.get(team_abbr, [])
    team_sk = skaters_df[skaters_df['team'] == team_abbr].copy()

    # Only care about OUT and IR — not day-to-day
    out_players = [p for p in injured if p['status'] in ('Out', 'Injured Reserve')]

    total_adj = 0.0
    impactful = []

    for p in out_players:
        name = p['name']
        last = name.split()[-1].lower()

        # Match to MoneyPuck by last name within team
        match = team_sk[team_sk['name'].str.lower().str.endswith(last)]
        if match.empty:
            # Try full league (traded players)
            match = skaters_df[skaters_df['name'].str.lower().str.contains(last, na=False)]

        if not match.empty:
            row = match.iloc[0]
            gs_pg      = row['gs_per_game']
            xg_pct     = row['onIce_xGoalsPercentage']
            ice_share  = row['icetime_per_game'] / 3600

            # Only flag players with meaningful impact (gs_per_game > 0.5)
            if gs_pg > 0.5:
                adj = -(xg_pct - 0.50) * ice_share * 0.35
                total_adj += adj
                impactful.append({
                    'name':    name,
                    'status':  p['status'],
                    'type':    p['type'],
                    'return':  p['return'],
                    'gs_pg':   round(gs_pg, 1),
                    'xg_pct':  round(xg_pct * 100, 1),
                    'adj':     round(adj * 100, 2),
                })
        else:
            # Unknown player — depth, minimal impact
            if p['status'] == 'Out':
                impactful.append({
                    'name':   name,
                    'status': p['status'],
                    'type':   p['type'],
                    'return': p['return'],
                    'gs_pg':  0,
                    'xg_pct': 50.0,
                    'adj':    0.0,
                })

    # Sort by impact
    impactful.sort(key=lambda x: x['gs_pg'], reverse=True)
    return total_adj, impactful

def load_skaters():
    df = pd.read_csv('data/skaters_mp.csv')
    df = df[df['situation'] == 'all']
    df = df[df['games_played'] >= 10]
    df['gs_per_game']      = df['gameScore'] / df['games_played']
    df['icetime_per_game'] = df['icetime']   / df['games_played']
    return df

def get_tonight_injuries(game_abbrs):
    """Main function — returns injury impact for all teams tonight."""
    print("  Fetching ESPN injuries...")
    injuries_by_team = get_all_injuries()
    skaters = load_skaters()

    impacts = {}
    for abbr in game_abbrs:
        adj, impactful = get_injury_impact(abbr, injuries_by_team, skaters)
        impacts[abbr] = {
            'adjustment': adj,
            'out':        impactful,
            'all_injured': injuries_by_team.get(abbr, [])
        }
    return impacts

if __name__ == '__main__':
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    data  = requests.get(f'https://api-web.nhle.com/v1/schedule/{today}').json()

    abbrs = set()
    for day in data.get('gameWeek', []):
        if day['date'] == today:
            for g in day.get('games', []):
                if g.get('gameState') not in ('OFF', 'FINAL'):
                    abbrs.add(g['homeTeam']['abbrev'])
                    abbrs.add(g['awayTeam']['abbrev'])

    skaters = load_skaters()
    injuries = get_all_injuries()

    print(f"\nInjury report for tonight's {len(abbrs)} teams:\n")
    for abbr in sorted(abbrs):
        adj, impactful = get_injury_impact(abbr, injuries, skaters)
        all_inj = injuries.get(abbr, [])
        out     = [p for p in all_inj if p['status'] in ('Out','Injured Reserve')]
        dtd     = [p for p in all_inj if p['status'] == 'Day-To-Day']

        if not all_inj:
            print(f"  {abbr}: clean")
            continue

        print(f"  {abbr}:")
        for p in out:
            gs = next((x['gs_pg'] for x in impactful if x['name']==p['name']), 0)
            gs_str = f" (gs/gm: {gs})" if gs > 0 else " (depth)"
            print(f"    OUT: {p['name']} — {p['type']} — back {p['return'] or 'unknown'}{gs_str}")
        for p in dtd:
            print(f"    DTD: {p['name']} — {p['type']}")
        if adj != 0:
            print(f"    Prob adjustment: {adj*100:+.1f}%")
        print()