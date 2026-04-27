import requests
import pandas as pd

print("Fetching NHL games...")

url = "https://api-web.nhle.com/v1/standings/now"
response = requests.get(url)
data = response.json()

# Pull the standings into a table
teams = data['standings']

rows = []
for team in teams:
    rows.append({
        'team': team['teamName']['default'],
        'wins': team['wins'],
        'losses': team['losses'],
        'ot_losses': team['otLosses'],
        'points': team['points'],
        'goals_for': team['goalFor'],
        'goals_against': team['goalAgainst'],
    })

df = pd.DataFrame(rows)
df = df.sort_values('points', ascending=False)

print(df.to_string(index=False))
df.to_csv('data/standings.csv', index=False)
print("\nSaved to data/standings.csv!")