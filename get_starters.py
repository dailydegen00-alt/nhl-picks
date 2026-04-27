import requests, re, pandas as pd

def get_todays_starters():
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    try:
        r = requests.get('https://www.dailyfaceoff.com/starting-goalies/', 
                        headers=headers, timeout=15)
        html = r.text

        all_names = re.findall(r'alt="([A-Z][a-z\-]+(?:\s+[A-Z][a-z\-\']+)+)"', html)
        skip_words = {'Line','Combination','Nation','Network','Daily','Faceoff',
                      'Money','Puck','Starting','Goalies','Hockey','Ltd'}
        goalie_names = [n for n in all_names
                        if not any(w in skip_words for w in n.split())
                        and len(n.split()) >= 2]

        mp = pd.read_csv('data/goalies_mp.csv')
        mp = mp[mp['situation'] == 'all']
        mp['gsax'] = mp['xGoals'] - mp['goals']

        starters = {}
        for goalie in goalie_names:
            last = goalie.split()[-1].lower()
            first_init = goalie.split()[0][0].lower()
            match = mp[mp['name'].str.split().str[-1].str.lower() == last]
            if len(match) == 1:
                row = match.iloc[0]
                starters[row['team']] = {'name': goalie, 'gsax': float(row['gsax'])}
            elif len(match) > 1:
                match2 = match[match['name'].str[0].str.lower() == first_init]
                if len(match2) >= 1:
                    row = match2.iloc[0]
                    starters[row['team']] = {'name': goalie, 'gsax': float(row['gsax'])}

        print(f"  DailyFaceoff: {len(starters)} starters found")
        for abbr, info in sorted(starters.items()):
            print(f"    {abbr}: {info['name']} (GSAx: {info['gsax']:.1f})")
        return starters

    except Exception as e:
        print(f"  DailyFaceoff error: {e}")
        return {}

if __name__ == '__main__':
    print("Today's confirmed starters:")
    starters = get_todays_starters()