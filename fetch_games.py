import requests
import pandas as pd

print("Fetching NHL game results...")

games = []

# Pull the last 3 months of games
for date in pd.date_range('2025-01-01', '2025-04-08'):
    url = f"https://api-web.nhle.com/v1/schedule/{date.strftime('%Y-%m-%d')}"
    data = requests.get(url).json()

    for day in data.get('gameWeek', []):
        for game in day.get('games', []):
            if game['gameState'] == 'OFF':  # only completed games
                games.append({
                    'date': day['date'],
                    'home_team': game['homeTeam']['placeName']['default'],
                    'away_team': game['awayTeam']['placeName']['default'],
                    'home_score': game['homeTeam']['score'],
                    'away_score': game['awayTeam']['score'],
                    'home_win': int(game['homeTeam']['score'] > game['awayTeam']['score'])
                })

df = pd.DataFrame(games).drop_duplicates()
df.to_csv('data/games.csv', index=False)
print(f"Done! Saved {len(df)} games to data/games.csv")