"""
Auto-grades yesterday's picks against NHL boxscores.
Tracks moneyline and O/U records separately.
"""
import requests
import json
import os
from datetime import datetime, timedelta, timezone

RECORD_FILE = 'data/record.json'

def load_record():
    if os.path.exists(RECORD_FILE):
        with open(RECORD_FILE) as f:
            return json.load(f)
    return {
        "alltime":   {"W": 0, "L": 0},
        "by_month":  {},
        "by_conf":   {
            "50": {"W":0,"L":0}, "55": {"W":0,"L":0},
            "60": {"W":0,"L":0}, "65": {"W":0,"L":0}, "70": {"W":0,"L":0}
        },
        "by_day": {},
        # O/U record
        "ou_alltime":  {"W": 0, "L": 0},
        "ou_by_month": {},
        "ou_by_conf":  {
            "50": {"W":0,"L":0}, "55": {"W":0,"L":0},
            "60": {"W":0,"L":0}, "65": {"W":0,"L":0}, "70": {"W":0,"L":0}
        },
    }

def save_record(r):
    with open(RECORD_FILE, 'w') as f:
        json.dump(r, f, indent=2)

def conf_band(c):
    if c >= 70: return "70"
    if c >= 65: return "65"
    if c >= 60: return "60"
    if c >= 55: return "55"
    return "50"

def get_results_for_date(date_str):
    results = {}
    try:
        data = requests.get(
            f'https://api-web.nhle.com/v1/schedule/{date_str}').json()
        for day in data.get('gameWeek', []):
            if day['date'] != date_str:
                continue
            for g in day.get('games', []):
                if g.get('gameState') not in ('OFF', 'FINAL'):
                    continue
                home_abbr  = g['homeTeam']['abbrev']
                away_abbr  = g['awayTeam']['abbrev']
                home_score = g['homeTeam'].get('score', 0)
                away_score = g['awayTeam'].get('score', 0)
                key = f"{away_abbr}@{home_abbr}"
                results[key] = {
                    'home_abbr':  home_abbr,
                    'away_abbr':  away_abbr,
                    'home_score': home_score,
                    'away_score': away_score,
                    'home_won':   home_score > away_score,
                    'total':      home_score + away_score,
                }
    except Exception as e:
        print(f"  Error fetching results for {date_str}: {e}")
    return results

def grade_day(date_str, record):
    day_data = record.get('by_day', {}).get(date_str, {})
    picks    = day_data.get('picks', [])

    if not picks:
        print(f"  No picks found for {date_str}")
        return record

    ml_already_graded = all(p.get('result') not in ('', None) for p in picks)
    prop_picks        = day_data.get('prop_picks', [])
    props_ungraded    = [pp for pp in prop_picks if pp.get('result') not in ('W', 'L')]

    if ml_already_graded and not props_ungraded:
        print(f"  {date_str} already graded")
        return record

    print(f"  Fetching results for {date_str}...")
    results = get_results_for_date(date_str)
    if not results:
        print(f"  No results found for {date_str}")
        return record

    ml_graded = 0
    ou_graded = 0

    # ── Grade ML + O/U ────────────────────────────────────────────────────────
    if not ml_already_graded:
        for pick in picks:
            home_abbr = pick['home_abbr']
            away_abbr = pick['away_abbr']
            key       = f"{away_abbr}@{home_abbr}"
            if key not in results:
                continue
            game = results[key]

            # Moneyline
            if pick.get('result') not in ('W', 'L'):
                pick_home = pick['pick_abbr'] == home_abbr
                home_won  = game['home_won']
                result    = 'W' if (pick_home == home_won) else 'L'
                pick['result']     = result
                pick['home_score'] = game['home_score']
                pick['away_score'] = game['away_score']

                band = conf_band(pick['conf'])
                record['alltime'][result] = record['alltime'].get(result, 0) + 1
                record.setdefault('by_conf', {}).setdefault(band, {"W":0,"L":0})
                record['by_conf'][band][result] = record['by_conf'][band].get(result, 0) + 1
                month = date_str[:7]
                record.setdefault('by_month', {}).setdefault(month, {"W":0,"L":0})
                record['by_month'][month][result] = record['by_month'][month].get(result, 0) + 1
                ml_graded += 1
                print(f"    ML  {away_abbr} @ {home_abbr} → Picked {pick['pick_abbr']} → {result} ({game['away_score']}-{game['home_score']})")

            # O/U
            ou_pick = pick.get('ou_pick')
            if ou_pick and pick.get('ou_result') not in ('W', 'L'):
                total     = game['total']
                ou_line   = pick.get('ou_line', 6.0)
                went_over = total > ou_line
                ou_result = 'W' if (
                    (ou_pick == 'Over'  and went_over) or
                    (ou_pick == 'Under' and not went_over)
                ) else 'L'

                pick['ou_result'] = ou_result
                pick['total']     = total

                ou_conf = pick.get('ou_conf', 55)
                band    = conf_band(ou_conf)
                record.setdefault('ou_alltime', {"W":0,"L":0})
                record['ou_alltime'][ou_result] = record['ou_alltime'].get(ou_result, 0) + 1
                record.setdefault('ou_by_conf', {}).setdefault(band, {"W":0,"L":0})
                record['ou_by_conf'][band][ou_result] = record['ou_by_conf'][band].get(ou_result, 0) + 1
                month = date_str[:7]
                record.setdefault('ou_by_month', {}).setdefault(month, {"W":0,"L":0})
                record['ou_by_month'][month][ou_result] = record['ou_by_month'][month].get(ou_result, 0) + 1
                ou_graded += 1
                print(f"    O/U {away_abbr} @ {home_abbr} → {ou_pick} {ou_line} → total {total} → {ou_result}")

        # Update day totals
        day_picks      = [p for p in picks if p.get('result') in ('W','L')]
        day_data['W']  = sum(1 for p in day_picks if p['result'] == 'W')
        day_data['L']  = sum(1 for p in day_picks if p['result'] == 'L')
        record['by_day'][date_str] = day_data
        print(f"  Graded {ml_graded} ML picks, {ou_graded} O/U picks for {date_str}")
    else:
        print(f"  ML/OU already graded — checking props only")

    # ── Grade props ────────────────────────────────────────────────────────────
    if not props_ungraded:
        return record

    print(f"  Props: {len(props_ungraded)} ungraded prop picks found")

    # Get game IDs for boxscore lookup
    game_ids = {}
    try:
        sched = requests.get(
            f'https://api-web.nhle.com/v1/schedule/{date_str}', timeout=10).json()
        for week in sched.get('gameWeek', []):
            if week['date'] != date_str:
                continue
            for g in week.get('games', []):
                if g.get('gameState') not in ('OFF', 'FINAL'):
                    continue
                ha = g['homeTeam']['abbrev']
                aa = g['awayTeam']['abbrev']
                game_ids[f"{aa}@{ha}"] = g['id']
    except Exception as e:
        print(f"  Props: schedule fetch error: {e}")
        return record

    print(f"  Props: found {len(game_ids)} completed games: {list(game_ids.keys())}")

    # Fetch boxscores
    player_stats = {}
    for game_key, game_id in game_ids.items():
        try:
            box = requests.get(
                f'https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore',
                timeout=10).json()

            # Player stats live under playerByGameStats, not homeTeam/awayTeam
            pbg = box.get('playerByGameStats', {})

            player_stats[game_key] = {}
            for side in ['homeTeam', 'awayTeam']:
                sd = pbg.get(side, {})

                # Skaters
                for cat in ['forwards', 'defense']:
                    for p in sd.get(cat, []):
                        # Name is {'default': 'C. Perry'} — abbreviated
                        name_raw = p.get('name', {}).get('default', '')
                        # Store both abbreviated and last-name-only versions
                        parts = name_raw.split(' ', 1)
                        last  = parts[1].strip().lower() if len(parts) > 1 else name_raw.lower()
                        full  = name_raw.strip().lower()
                        stats = {
                            'shots':   p.get('sog', 0),   # shots on goal
                            'goals':   p.get('goals', 0),
                            'assists': p.get('assists', 0),
                            'points':  p.get('points', 0),
                        }
                        player_stats[game_key][full] = stats
                        player_stats[game_key][last] = stats  # also index by last name

                # Goalies
                for p in sd.get('goalies', []):
                    name_raw = p.get('name', {}).get('default', '')
                    parts = name_raw.split(' ', 1)
                    last  = parts[1].strip().lower() if len(parts) > 1 else name_raw.lower()
                    full  = name_raw.strip().lower()
                    sa = p.get('shotsAgainst', p.get('saveShotsAgainst', 0))
                    ga = p.get('goalsAgainst', 0)
                    sv = p.get('saves', max(sa - ga, 0))
                    stats = {'saves': sv, 'shots': sv}
                    player_stats[game_key][full] = stats
                    player_stats[game_key][last] = stats

        except Exception as e:
            print(f"  Props: boxscore error {game_key}: {e}")

    if not player_stats:
        print(f"  Props: no player stats fetched — boxscores may not be available yet")
        return record

    print(f"  Props: player stats loaded for {list(player_stats.keys())}")

    from difflib import get_close_matches
    prop_graded = 0

    for pp in props_ungraded:
        ha       = pp.get('home_abbr', '')
        aa       = pp.get('away_abbr', '')
        game_key = f"{aa}@{ha}"
        stats    = player_stats.get(game_key, {})
        if not stats:
            continue

        player = pp['player'].lower()
        market = pp['market']
        pick   = pp['pick']
        line   = float(pp.get('line', 0))

        matches = get_close_matches(player, stats.keys(), n=1, cutoff=0.72)
        if not matches:
            # Try last name only
            last = player.split()[-1] if player.split() else player
            matches = get_close_matches(last, stats.keys(), n=1, cutoff=0.80)
        if not matches:
            continue
        pstats = stats[matches[0]]

        if market == 'player_shots_on_goal':
            actual = pstats.get('shots', 0)
        elif market == 'player_goalie_saves':
            actual = pstats.get('saves', 0)
        elif market in ('player_points', 'player_points_assists'):
            actual = pstats.get('points', 0)
        else:
            continue

        result = 'W' if (
            ('Over'  in pick and actual > line) or
            ('Under' in pick and actual < line)
        ) else 'L'

        pp['result'] = result
        pp['actual'] = actual

        conf  = pp.get('conf', 55)
        band  = conf_band(conf)
        month = date_str[:7]

        record.setdefault('prop_alltime', {"W":0,"L":0})
        record['prop_alltime'][result] = record['prop_alltime'].get(result, 0) + 1
        record.setdefault('prop_by_conf', {}).setdefault(band, {"W":0,"L":0})
        record['prop_by_conf'][band][result] = record['prop_by_conf'][band].get(result, 0) + 1
        record.setdefault('prop_by_month', {}).setdefault(month, {"W":0,"L":0})
        record['prop_by_month'][month][result] = record['prop_by_month'][month].get(result, 0) + 1

        prop_graded += 1
        print(f"    PROP {pp['player']} {pick} → actual {actual:.0f} → {result}")

    if prop_graded:
        print(f"  Graded {prop_graded} prop picks for {date_str}")

    record['by_day'][date_str] = day_data
    return record
    return record

def grade_recent(days=3):
    record = load_record()
    now    = datetime.now(timezone.utc)
    for i in range(1, days + 1):
        date_str = (now - timedelta(days=i)).strftime('%Y-%m-%d')
        print(f"\nGrading {date_str}...")
        record = grade_day(date_str, record)
    save_record(record)
    print("\nRecord saved.")
    return record

if __name__ == '__main__':
    print("Auto-grading recent picks...")
    rec = grade_recent(days=5)

    at    = rec['alltime']
    total = at.get('W',0) + at.get('L',0)
    pct   = at['W']/total*100 if total else 0
    print(f"\nMoneyline all-time: {at.get('W',0)}-{at.get('L',0)} ({pct:.1f}%)")

    ou_at  = rec.get('ou_alltime', {"W":0,"L":0})
    ou_tot = ou_at.get('W',0) + ou_at.get('L',0)
    ou_pct = ou_at['W']/ou_tot*100 if ou_tot else 0
    print(f"O/U all-time:       {ou_at.get('W',0)}-{ou_at.get('L',0)} ({ou_pct:.1f}%)")

    print("\nML by confidence:")
    for band in ['50','55','60','65','70']:
        b = rec['by_conf'].get(band, {"W":0,"L":0})
        t = b['W']+b['L']
        p = b['W']/t*100 if t else 0
        label = {'50':'50-55%','55':'55-60%','60':'60-65%','65':'65-70%','70':'70%+'}.get(band)
        print(f"  {label}: {b['W']}-{b['L']} ({p:.0f}%)")

    print("\nO/U by confidence:")
    for band in ['50','55','60','65','70']:
        b = rec.get('ou_by_conf',{}).get(band, {"W":0,"L":0})
        t = b['W']+b['L']
        p = b['W']/t*100 if t else 0
        label = {'50':'50-55%','55':'55-60%','60':'60-65%','65':'65-70%','70':'70%+'}.get(band)
        print(f"  {label}: {b['W']}-{b['L']} ({p:.0f}%)")