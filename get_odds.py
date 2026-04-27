import requests

ODDS_API_KEY = 'a62a46c2a2679e8ce805ecf918c948c0'

def decimal_to_american(dec):
    if dec >= 2.0:
        return f"+{int((dec-1)*100)}"
    else:
        return str(int(-(1/(dec-1))*100))

def decimal_to_implied(dec):
    return 1 / dec

def get_ev(model_prob, decimal_odds):
    win_payout = (decimal_odds - 1) * 100
    ev = (model_prob * win_payout) - ((1 - model_prob) * 100)
    return round(ev, 1)

def get_tonights_odds():
    """Fetch NHL moneyline AND totals odds."""
    url = (f'https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds/'
           f'?apiKey={ODDS_API_KEY}&regions=us&markets=h2h,totals&bookmakers=draftkings')
    try:
        r    = requests.get(url, timeout=10)
        data = r.json()
        if not isinstance(data, list):
            print(f"  Odds API error: {data}")
            return {}

        odds_map = {}
        for game in data:
            home = game.get('home_team','')
            away = game.get('away_team','')
            entry = {'away_team': away}

            for bk in game.get('bookmakers',[]):
                for mkt in bk.get('markets',[]):

                    # ── Moneyline ─────────────────────────
                    if mkt['key'] == 'h2h':
                        prices = {o['name']: o['price'] for o in mkt['outcomes']}
                        if home in prices and away in prices:
                            h_dec = prices[home]
                            a_dec = prices[away]
                            # skip obviously finished games
                            if h_dec < 1.01 or a_dec < 1.01:
                                continue
                            h_imp   = decimal_to_implied(h_dec)
                            a_imp   = decimal_to_implied(a_dec)
                            total   = h_imp + a_imp
                            entry.update({
                                'home_odds_dec': h_dec,
                                'away_odds_dec': a_dec,
                                'home_odds_am':  decimal_to_american(h_dec),
                                'away_odds_am':  decimal_to_american(a_dec),
                                'home_implied':  h_imp / total,
                                'away_implied':  a_imp / total,
                            })

                    # ── Totals ────────────────────────────
                    elif mkt['key'] == 'totals':
                        over  = next((o for o in mkt['outcomes'] if o['name']=='Over'),  None)
                        under = next((o for o in mkt['outcomes'] if o['name']=='Under'), None)
                        if over and under:
                            line      = over.get('point', 6.0)
                            over_dec  = over['price']
                            under_dec = under['price']
                            # skip bad odds
                            if over_dec < 1.01 or under_dec < 1.01:
                                continue
                            o_imp = decimal_to_implied(over_dec)
                            u_imp = decimal_to_implied(under_dec)
                            tot   = o_imp + u_imp
                            entry.update({
                                'ou_line':         line,
                                'over_dec':        over_dec,
                                'under_dec':       under_dec,
                                'over_am':         decimal_to_american(over_dec),
                                'under_am':        decimal_to_american(under_dec),
                                'over_implied':    o_imp / tot,
                                'under_implied':   u_imp / tot,
                            })

            if 'home_implied' in entry or 'ou_line' in entry:
                odds_map[home] = entry

        print(f"  Odds: {len(odds_map)} games found")
        return odds_map

    except Exception as e:
        print(f"  Odds error: {e}")
        return {}

if __name__ == '__main__':
    odds = get_tonights_odds()
    for home, info in odds.items():
        print(f"\n{info['away_team']} @ {home}")
        if 'home_odds_am' in info:
            print(f"  ML:  {home} {info['home_odds_am']} · {info['away_team']} {info['away_odds_am']}")
            print(f"       Implied: {info['home_implied']:.0%} / {info['away_implied']:.0%}")
        if 'ou_line' in info:
            print(f"  O/U: {info['ou_line']} — Over {info['over_am']} / Under {info['under_am']}")
            print(f"       Implied: Over {info['over_implied']:.0%} / Under {info['under_implied']:.0%}")