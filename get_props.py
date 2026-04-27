import requests
from difflib import get_close_matches

ODDS_API_KEY = 'a62a46c2a2679e8ce805ecf918c948c0'

PROP_MARKETS = [
    'player_shots_on_goal',
    'player_goalie_saves',
    'player_points',
    'player_points_assists',
]

PROP_LABELS = {
    'player_shots_on_goal':   'Shots on Goal',
    'player_goalie_saves':    'Saves',
    'player_points':          'Points',
    'player_points_assists':  'Points + Assists',
}

def decimal_to_american(dec):
    if dec >= 2.0:
        return f"+{int((dec-1)*100)}"
    return str(int(-(1/(dec-1))*100))

def decimal_to_implied(dec):
    return round(1/dec, 4)

def get_nhl_events():
    """Get today's NHL event IDs from The Odds API."""
    url = f'https://api.the-odds-api.com/v4/sports/icehockey_nhl/events?apiKey={ODDS_API_KEY}'
    try:
        r = requests.get(url, timeout=10)
        return r.json() if isinstance(r.json(), list) else []
    except Exception as e:
        print(f"  Events error: {e}")
        return []

def discover_valid_markets(event_id):
    """Try each market individually to find which ones are valid."""
    candidates = [
        'player_anytime_goal_scorer',
        'player_goal_scorer',
        'player_points',
        'player_points_assists',
        'player_shots_on_goal',
        'player_goalie_saves',
        'player_saves',
        'player_goal',
        'batter_home_runs',  # just to confirm invalid returns error
    ]
    valid = []
    for m in candidates:
        url = (f'https://api.the-odds-api.com/v4/sports/icehockey_nhl/events/{event_id}/odds'
               f'?apiKey={ODDS_API_KEY}&regions=us&markets={m}&bookmakers=draftkings&oddsFormat=decimal')
        try:
            r = requests.get(url, timeout=10).json()
            if 'error_code' in r:
                print(f"  INVALID: {m}")
            elif r.get('bookmakers'):
                print(f"  VALID + HAS DATA: {m}")
                valid.append(m)
            else:
                print(f"  VALID but no data: {m}")
                valid.append(m)
        except Exception as e:
            print(f"  ERROR {m}: {e}")
    return valid

def get_props_for_event(event_id):
    """Fetch all prop markets for a single event, one market at a time to avoid invalid market errors."""
    all_bookmakers = {}
    for market in PROP_MARKETS:
        url = (f'https://api.the-odds-api.com/v4/sports/icehockey_nhl/events/{event_id}/odds'
               f'?apiKey={ODDS_API_KEY}&regions=us&markets={market}&bookmakers=draftkings&oddsFormat=decimal')
        try:
            r = requests.get(url, timeout=10).json()
            if 'error_code' in r:
                continue
            for bk in r.get('bookmakers', []):
                key = bk['key']
                if key not in all_bookmakers:
                    all_bookmakers[key] = {'key': key, 'markets': []}
                all_bookmakers[key]['markets'].extend(bk.get('markets', []))
        except Exception as e:
            print(f"  Props error {market} for {event_id}: {e}")
    return {'bookmakers': list(all_bookmakers.values())}

def get_all_props():
    """
    Returns a dict keyed by home_team name:
    {
      'Boston Bruins': {
        'away': 'New Jersey Devils',
        'props': [
          {
            'player':   'David Pastrnak',
            'market':   'player_goal_scorer',
            'label':    'Anytime Goal',
            'type':     'binary',       # yes/no
            'line':     None,
            'over_dec': 2.50,
            'over_am':  '+150',
            'over_imp': 0.40,
          },
          {
            'player':   'David Pastrnak',
            'market':   'player_shots_on_goal',
            'label':    'Shots on Goal',
            'type':     'ou',           # over/under
            'line':     3.5,
            'over_dec': 1.87,
            'over_am':  '-115',
            'over_imp': 0.535,
            'under_dec':1.95,
            'under_am': '-105',
            'under_imp':0.513,
          },
        ]
      }
    }
    """
    events = get_nhl_events()
    if not events:
        print("  No events found for props")
        return {}

    print(f"  Found {len(events)} events for props")
    all_props = {}

    for event in events:
        home = event.get('home_team', '')
        away = event.get('away_team', '')
        event_id = event.get('id', '')
        if not event_id:
            continue

        data = get_props_for_event(event_id)
        if not data or not data.get('bookmakers'):
            continue

        props_list = []
        for bk in data.get('bookmakers', []):
            if bk.get('key') != 'draftkings':
                continue
            for mkt in bk.get('markets', []):
                market_key = mkt.get('key', '')
                label = PROP_LABELS.get(market_key, market_key)

                # All valid markets are O/U style
                grouped = {}
                for o in mkt.get('outcomes', []):
                    player = o.get('description', '')
                    side   = o.get('name', '')
                    line   = o.get('point', 0)
                    dec    = o.get('price', 0)
                    if dec < 1.01 or not player: continue
                    key = (player, line)
                    if key not in grouped:
                        grouped[key] = {'player': player, 'line': line}
                    if side == 'Over':
                        grouped[key]['over_dec'] = dec
                        grouped[key]['over_am']  = decimal_to_american(dec)
                        grouped[key]['over_imp'] = decimal_to_implied(dec)
                    elif side == 'Under':
                        grouped[key]['under_dec'] = dec
                        grouped[key]['under_am']  = decimal_to_american(dec)
                        grouped[key]['under_imp'] = decimal_to_implied(dec)

                for (player, line), entry in grouped.items():
                    if 'over_dec' not in entry: continue
                    props_list.append({
                        **entry,
                        'market': market_key,
                        'label':  label,
                        'type':   'ou',
                    })

        if props_list:
            all_props[home] = {'away': away, 'props': props_list}

    print(f"  Props: {sum(len(v['props']) for v in all_props.values())} props across {len(all_props)} games")
    return all_props


def fuzzy_match_player(name, player_index, cutoff=0.72):
    """Match an Odds API player name to a MoneyPuck player name."""
    matches = get_close_matches(name.lower(), [p.lower() for p in player_index], n=1, cutoff=cutoff)
    if not matches:
        return None
    # Return the original-cased version
    lower_to_orig = {p.lower(): p for p in player_index}
    return lower_to_orig.get(matches[0])


if __name__ == '__main__':
    props = get_all_props()
    for home, data in list(props.items())[:2]:
        print(f"\n{data['away']} @ {home}")
        for p in data['props'][:5]:
            if p['type'] == 'binary':
                print(f"  {p['label']:20s} {p['player']:25s} Yes {p['over_am']} (imp {p['over_imp']:.0%})")
            else:
                print(f"  {p['label']:20s} {p['player']:25s} O/U {p['line']} · Over {p['over_am']} (imp {p['over_imp']:.0%})")