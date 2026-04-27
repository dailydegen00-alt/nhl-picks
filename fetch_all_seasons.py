import requests
import pandas as pd

def fetch_season(start, end, label):
    print(f"Fetching {label}...")
    games = []
    for date in pd.date_range(start, end):
        url = f'https://api-web.nhle.com/v1/schedule/{date.strftime("%Y-%m-%d")}'
        data = requests.get(url).json()
        for day in data.get('gameWeek', []):
            for g in day.get('games', []):
                if g['gameState'] == 'OFF':
                    games.append({
                        'date':       day['date'],
                        'home_team':  g['homeTeam']['placeName']['default'],
                        'away_team':  g['awayTeam']['placeName']['default'],
                        'home_score': g['homeTeam']['score'],
                        'away_score': g['awayTeam']['score'],
                        'home_win':   int(g['homeTeam']['score'] > g['awayTeam']['score'])
                    })
    df = pd.DataFrame(games).drop_duplicates()
    print(f"  Got {len(df)} games")
    return df

# 2023-24 season
df_2324 = fetch_season('2023-10-01', '2024-04-30', '2023-24')
df_2324.to_csv('data/games_2324.csv', index=False)

# 2024-25 season
df_2425 = fetch_season('2024-10-01', '2025-04-30', '2024-25')
df_2425.to_csv('data/games_2425.csv', index=False)

print("\nDone!")
print(f"2023-24: {len(df_2324)} games")
print(f"2024-25: {len(df_2425)} games")