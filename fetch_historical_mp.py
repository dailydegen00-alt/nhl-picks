import requests
import pandas as pd

headers = {'User-Agent': 'Mozilla/5.0'}

seasons = [
    (2024, 'data/mp_2024.csv'),  # 2024-25 season
    (2023, 'data/mp_2023.csv'),  # 2023-24 season
]

for year, path in seasons:
    print(f"Fetching MoneyPuck {year}-{year+1} season...")
    url = f'https://moneypuck.com/moneypuck/playerData/seasonSummary/{year}/regular/teams.csv'
    r = requests.get(url, headers=headers)
    with open(path, 'wb') as f:
        f.write(r.content)
    df = pd.read_csv(path)
    df = df[df['situation'] == '5on5']
    print(f"  Got {len(df)} team records")
    print(f"  Sample teams: {df['name'].tolist()[:5]}")
    print()

print("Done!")