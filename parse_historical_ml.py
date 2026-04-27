"""
parse_historical_ml.py (v2)
Parses tab-delimited odds from checkbestodds.com saved as raw_odds.txt
Output: data/historical_ml_odds.csv
"""
import pandas as pd
import re
import os

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

def to_abbr(name):
    name = name.strip()
    if name in FULL_TO_ABBR: return FULL_TO_ABBR[name]
    for full, abbr in FULL_TO_ABBR.items():
        if name.lower() == full.lower(): return abbr
    return name[:3].upper()

def parse_date(raw):
    months = {'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,
              'July':7,'August':8,'September':9,'October':10,'November':11,'December':12}
    parts = raw.strip().split()
    if len(parts) == 3:
        try:
            day = int(parts[0]); mon = months.get(parts[1], 0); yr = int(parts[2])
            if mon: return f'{yr}-{mon:02d}-{day:02d}'
        except: pass
    return None

raw_path = 'data/raw_odds.txt'
if not os.path.exists(raw_path):
    print(f"ERROR: {raw_path} not found.")
    exit()

with open(raw_path, 'r', encoding='utf-8') as f:
    lines = [l.rstrip('\n') for l in f.readlines()]

print(f"Read {len(lines)} lines")

rows = []
current_date = None
date_pat = re.compile(r'^\d{1,2}\s+\w+\s+\d{4}')

for line in lines:
    # Split on tabs
    parts = line.split('\t')

    # Date header: "09 October 2025\t1\tX\t2"
    if len(parts) >= 1 and date_pat.match(parts[0]):
        current_date = parse_date(parts[0])
        continue

    # Game line: "16:00 Home Team - Away Team\t2.60\t4.20\t2.10"
    if len(parts) >= 4 and current_date:
        game_part = parts[0].strip()
        try:
            home_odds = float(parts[1])
            draw_odds = float(parts[2])
            away_odds = float(parts[3])
        except:
            continue

        # Parse time + teams from game_part
        # Format: "16:00 Home Team - Away Team"
        time_match = re.match(r'^\d{2}:\d{2}\s+(.+?)\s+-\s+(.+)$', game_part)
        if not time_match:
            continue

        home_full = time_match.group(1).strip()
        away_full = time_match.group(2).strip()

        margin    = 1/home_odds + 1/draw_odds + 1/away_odds
        home_imp  = round((1/home_odds) / margin * 100, 1)
        away_imp  = round((1/away_odds) / margin * 100, 1)

        rows.append({
            'date':           current_date,
            'home_team':      to_abbr(home_full),
            'away_team':      to_abbr(away_full),
            'home_team_full': home_full,
            'away_team_full': away_full,
            'home_odds_dec':  home_odds,
            'draw_odds_dec':  draw_odds,
            'away_odds_dec':  away_odds,
            'home_implied':   home_imp,
            'away_implied':   away_imp,
        })

print(f"Parsed {len(rows)} games")
if not rows:
    print("No games parsed.")
    exit()

df = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
print(f"Date range: {df['date'].min()} → {df['date'].max()}")
print(f"Sample:\n{df.head(3)[['date','home_team','away_team','home_odds_dec','away_odds_dec']].to_string(index=False)}")

df.to_csv('data/historical_ml_odds.csv', index=False)
print(f"\n✅ Saved data/historical_ml_odds.csv ({len(df)} games)")
print("Next: python backtest.py")