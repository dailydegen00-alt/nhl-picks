import requests
import pandas as pd
import joblib
from scipy.stats import norm, poisson
import json
import os
from datetime import datetime, timedelta, timezone
from get_starters import get_todays_starters
from get_injuries import load_skaters, get_all_injuries, get_injury_impact
from grade import load_record, save_record, grade_recent
from get_odds import get_tonights_odds, get_ev
from get_props import get_all_props, fuzzy_match_player
from build_historical_features import load_mp

# ── Load models ───────────────────────────────
saved       = joblib.load('data/model.pkl')
model       = saved['model']
features    = saved['features']

saved_ou     = joblib.load('data/model_total.pkl')
model_ou     = saved_ou['model']
features_ou  = saved_ou['features']
residual_std = saved_ou['residual_std']
default_line = 6.0

abbr_to_full = {
    'ANA':'Anaheim Ducks','BOS':'Boston Bruins','BUF':'Buffalo Sabres',
    'CGY':'Calgary Flames','CAR':'Carolina Hurricanes','CHI':'Chicago Blackhawks',
    'COL':'Colorado Avalanche','CBJ':'Columbus Blue Jackets','DAL':'Dallas Stars',
    'DET':'Detroit Red Wings','EDM':'Edmonton Oilers','FLA':'Florida Panthers',
    'LAK':'Los Angeles Kings','MIN':'Minnesota Wild','MTL':'Montréal Canadiens',
    'NSH':'Nashville Predators','NJD':'New Jersey Devils','NYR':'New York Rangers',
    'NYI':'New York Islanders','OTT':'Ottawa Senators','PHI':'Philadelphia Flyers',
    'PIT':'Pittsburgh Penguins','SJS':'San Jose Sharks','SEA':'Seattle Kraken',
    'STL':'St. Louis Blues','TBL':'Tampa Bay Lightning','TOR':'Toronto Maple Leafs',
    'UTA':'Utah Mammoth','VAN':'Vancouver Canucks','VGK':'Vegas Golden Knights',
    'WSH':'Washington Capitals','WPG':'Winnipeg Jets'
}

# ── Auto-grade ────────────────────────────────
print("Auto-grading recent picks...")
record = grade_recent(days=5)

# ── Sync actual scores to historical_lines.csv ────────────────────────────────
lines_path = 'data/historical_lines.csv'
if os.path.exists(lines_path):
    lines_df = pd.read_csv(lines_path)
    updated  = 0
    for date_str, day_data in record.get('by_day', {}).items():
        for p in day_data.get('picks', []):
            hs = p.get('home_score', 0)
            as_ = p.get('away_score', 0)
            if not hs and not as_: continue  # not graded yet
            mask = (
                (lines_df['date'] == date_str) &
                (lines_df['home_team'] == p.get('home_abbr','')) &
                (lines_df['away_team'] == p.get('away_abbr',''))
            )
            if mask.any() and lines_df.loc[mask, 'actual_total'].iloc[0] == '':
                lines_df.loc[mask, 'home_score']   = hs
                lines_df.loc[mask, 'away_score']   = as_
                lines_df.loc[mask, 'actual_total'] = hs + as_
                updated += 1
    if updated:
        lines_df.to_csv(lines_path, index=False)
        print(f"  Updated {updated} game results in historical_lines.csv")

# ── Data ──────────────────────────────────────
team_stats = load_mp('data/moneypuck.csv')

print("Fetching standings...")
stand_raw = requests.get('https://api-web.nhle.com/v1/standings/now').json()['standings']
standings = {}
for t in stand_raw:
    abbr = t['teamAbbrev']['default']
    standings[abbr] = {
        'points':     t['points'],
        'gp':         t['gamesPlayed'],
        'home_wins':  t['homeWins'],
        'home_losses':t['homeLosses']+t['homeOtLosses'],
        'road_wins':  t['roadWins'],
        'road_losses':t['roadLosses']+t['roadOtLosses'],
        'l10w':       t['l10Wins'],
        'l10l':       t['l10Losses'],
        'wildcard':   t.get('wildcardSequence',99),
        'div_rank':   t.get('divisionSequence',99),
        'clinch':     t.get('clinchIndicator',''),
    }

print("Fetching starters...")
starters = get_todays_starters()
print("Fetching injuries...")
all_injuries = get_all_injuries()
skaters_df   = load_skaters()
print("Fetching odds...")
odds_map = get_tonights_odds()
print("Fetching props...")
props_map = get_all_props()

# ── Load player stats for model edge ─────────
print("Loading player stats...")
print(f"Loading player stats...")
try:
    skaters_prop = pd.read_csv('data/skaters_mp.csv')
    skaters_prop = skaters_prop[skaters_prop['situation'] == 'all'].copy()
    skaters_prop['gpg']    = skaters_prop['I_F_goals']   / skaters_prop['games_played'].clip(lower=1)
    skaters_prop['ptspg']  = (skaters_prop['I_F_goals'] + skaters_prop['I_F_primaryAssists'] + skaters_prop['I_F_secondaryAssists']) / skaters_prop['games_played'].clip(lower=1)
    skaters_prop['shotspg']= skaters_prop['I_F_shotsOnGoal'] / skaters_prop['games_played'].clip(lower=1)
    skaters_prop = skaters_prop.dropna(subset=['name'])

    # Build opponent shots-allowed lookup for prop adjustment
    _mp_all = pd.read_csv('data/moneypuck.csv')
    _mp_all = _mp_all[_mp_all['situation'] == 'all'].copy()
    _mp_all['soga_pg'] = _mp_all['shotsOnGoalAgainst'] / _mp_all['games_played'].clip(lower=1)
    _league_avg_soga = _mp_all['soga_pg'].mean()
    _opp_soga = dict(zip(_mp_all['name'], _mp_all['soga_pg']))

    skaters_prop_index = skaters_prop['name'].tolist()
except Exception as e:
    print(f"  Skater stats error: {e}")
    skaters_prop = pd.DataFrame()
    skaters_prop_index = []

try:
    goalies_prop = pd.read_csv('data/goalies_mp.csv')
    goalies_prop = goalies_prop[goalies_prop['situation'] == 'all'].copy()
    # saves = shots on goal faced - goals allowed
    goalies_prop['savespg'] = (goalies_prop['ongoal'] - goalies_prop['goals']) / goalies_prop['games_played'].clip(lower=1)
    goalies_prop = goalies_prop.dropna(subset=['name'])
    goalies_prop_index = goalies_prop['name'].tolist()
except Exception as e:
    print(f"  Goalie stats error: {e}")
    goalies_prop = pd.DataFrame()
    goalies_prop_index = []

def get_importance(hs, as_):
    if hs.get('clinch')=='e' and as_.get('clinch')=='e': return 1,'Both Eliminated'
    if hs.get('clinch')=='e': return 2,'Home Eliminated'
    if as_.get('clinch')=='e': return 2,'Away Eliminated'
    gl  = max(82-hs.get('gp',82),0)
    hir = hs.get('wildcard',99)<=5 or hs.get('div_rank',99)<=4
    air = as_.get('wildcard',99)<=5 or as_.get('div_rank',99)<=4
    if gl<=5  and hir and air: return 10,'Must Win — Both In Playoff Race'
    if gl<=5  and (hir or air): return 8,'High Stakes — Playoff Race'
    if gl<=15 and hir and air: return 7,'Both Teams In Playoff Hunt'
    if hir or air: return 5,'One Team With Playoff Stakes'
    return 3,'Regular Season'

def get_rest_days(abbr):
    now = datetime.now(timezone.utc)
    for i in range(1,8):
        d = (now-timedelta(days=i)).strftime('%Y-%m-%d')
        try:
            data = requests.get(f'https://api-web.nhle.com/v1/schedule/{d}').json()
            for gw in data.get('gameWeek',[]):
                for g in gw.get('games',[]):
                    if g.get('gameState') not in ('OFF','FINAL'): continue
                    if g['homeTeam']['abbrev']==abbr or g['awayTeam']['abbrev']==abbr:
                        return i-1
        except: pass
    return 7

def home_road_rate(s, is_home):
    if is_home: w,l = s.get('home_wins',0),s.get('home_losses',0)
    else:        w,l = s.get('road_wins',0),s.get('road_losses',0)
    return w/max(w+l,1)

# ── Tonight's games ───────────────────────────
# Use Pacific time (PST = UTC-7)
pst_now       = datetime.now(timezone.utc) - timedelta(hours=7)
today         = pst_now.strftime('%Y-%m-%d')
today_display = pst_now.strftime('%A, %B %d %Y')
sched         = requests.get(f'https://api-web.nhle.com/v1/schedule/{today}').json()

tonight = []
for day in sched.get('gameWeek',[]):
    if day['date']==today:
        for g in day.get('games',[]):
            if g.get('gameState') in ('OFF','FINAL','CRIT'): continue
            ha = g['homeTeam']['abbrev']
            aa = g['awayTeam']['abbrev']
            hf = abbr_to_full.get(ha)
            af = abbr_to_full.get(aa)
            if hf and af:
                tonight.append({'home_abbr':ha,'away_abbr':aa,'home':hf,'away':af})

print(f"Found {len(tonight)} games")

print("Checking rest days...")
rest = {}
for g in tonight:
    for abbr in [g['home_abbr'],g['away_abbr']]:
        if abbr not in rest:
            rest[abbr] = get_rest_days(abbr)

# ── Build picks ───────────────────────────────
picks = []
for game in tonight:
    hf,af  = game['home'],game['away']
    ha,aa  = game['home_abbr'],game['away_abbr']
    if hf not in team_stats.index or af not in team_stats.index: continue

    def gs(team, col):
        try: return float(team_stats.loc[team,col])
        except: return 0.5

    # ML row
    row = {}
    for feat in features:
        if feat.startswith('home_'):   row[feat] = gs(hf, feat[5:])
        elif feat.startswith('away_'): row[feat] = gs(af, feat[5:])
        elif feat.endswith('_diff'):   row[feat] = gs(hf,feat[:-5]) - gs(af,feat[:-5])
        else:                          row[feat] = 0

    try:
        prob = model.predict_proba(pd.DataFrame([row]))[0][1]
    except Exception as e:
        print(f"  SKIPPING {hf} vs {af}: {e}")
        continue

    hs  = standings.get(ha,{})
    as_ = standings.get(aa,{})

    prob = min(0.95,max(0.05,prob+(home_road_rate(hs,True)-0.5)*0.08-(home_road_rate(as_,False)-0.5)*0.08))
    h_rest = rest.get(ha,7); a_rest = rest.get(aa,7)
    if h_rest==0 and a_rest>0: prob-=0.04
    elif a_rest==0 and h_rest>0: prob+=0.04
    prob = min(0.95,max(0.05,prob+(hs.get('l10w',5)/10-as_.get('l10w',5)/10)*0.06))
    _,imp_label = get_importance(hs,as_)
    if hs.get('clinch')=='e' and as_.get('clinch')!='e': prob-=0.06
    elif as_.get('clinch')=='e' and hs.get('clinch')!='e': prob+=0.06

    h_gsax = starters.get(ha,{}).get('gsax',0)
    a_gsax = starters.get(aa,{}).get('gsax',0)
    prob   = min(0.95,max(0.05,prob+(h_gsax-a_gsax)*0.002))

    h_inj_adj,h_injured = get_injury_impact(ha,all_injuries,skaters_df)
    a_inj_adj,a_injured = get_injury_impact(aa,all_injuries,skaters_df)
    prob = min(0.95,max(0.05,prob+h_inj_adj-a_inj_adj))

    if prob>=0.5: winner,conf,pa=hf,prob,ha
    else:         winner,conf,pa=af,1-prob,aa

    # O/U row
    ou_row = {'gsax_diff': h_gsax - a_gsax}
    for feat in features_ou:
        if feat.startswith('home_'):   ou_row[feat] = gs(hf,feat[5:])
        elif feat.startswith('away_'): ou_row[feat] = gs(af,feat[5:])
        elif feat.endswith('_diff'):   ou_row[feat] = gs(hf,feat[:-5])-gs(af,feat[:-5])
        else:                          ou_row[feat] = 0

    go_pre    = odds_map.get(hf,{})
    book_line = go_pre.get('ou_line', default_line)
    try:
        pred_total = float(model_ou.predict(pd.DataFrame([ou_row]))[0])
        over_prob  = float(1 - norm.cdf(book_line, loc=pred_total, scale=residual_std))
        print(f"  O/U {af} @ {hf}: pred={pred_total:.2f} line={book_line} over_prob={over_prob:.2f}")
    except:
        over_prob = 0.5

    # Goalie adjustment
    gsax_sum = h_gsax + a_gsax
    if gsax_sum > 40:   over_prob -= 0.05
    elif gsax_sum > 20: over_prob -= 0.02
    elif gsax_sum < 0:  over_prob += 0.03

    over_prob  = min(0.95, max(0.05, over_prob))
    ou_pick    = 'Over' if over_prob >= 0.5 else 'Under'
    ou_conf    = over_prob if ou_pick=='Over' else 1-over_prob

    # ML odds
    go = odds_map.get(hf,{})
    pih = winner==hf
    book_imp=ml_ev=ml_am=ml_dec=None
    if go and 'home_implied' in go:
        if pih: book_imp=go.get('home_implied'); ml_dec=go.get('home_odds_dec'); ml_am=go.get('home_odds_am')
        else:   book_imp=go.get('away_implied'); ml_dec=go.get('away_odds_dec'); ml_am=go.get('away_odds_am')
        if ml_dec: ml_ev = get_ev(conf,ml_dec)

    # O/U odds
    ou_line = go.get('ou_line',default_line) if go else default_line
    ou_bimp=ou_ev=ou_am=ou_dec=None
    if go and 'ou_line' in go:
        if ou_pick=='Over': ou_bimp=go.get('over_implied'); ou_dec=go.get('over_dec'); ou_am=go.get('over_am')
        else:               ou_bimp=go.get('under_implied'); ou_dec=go.get('under_dec'); ou_am=go.get('under_am')
        if ou_dec: ou_ev = get_ev(ou_conf,ou_dec)

    if   conf>=0.65: tier='strong'
    elif conf>=0.60: tier='lean'
    elif conf>=0.55: tier='lean'
    else:            tier='skip'

    picks.append({
        'home':hf,'away':af,'home_abbr':ha,'away_abbr':aa,
        'winner':winner,'pick_abbr':pa,'confidence':conf,'tier':tier,
        'imp_label':imp_label,
        'h_goalie':starters.get(ha,{}).get('name','TBD'),
        'a_goalie':starters.get(aa,{}).get('name','TBD'),
        'h_gsax':h_gsax,'a_gsax':a_gsax,
        'h_l10w':hs.get('l10w','?'),'h_l10l':hs.get('l10l','?'),
        'a_l10w':as_.get('l10w','?'),'a_l10l':as_.get('l10l','?'),
        'h_rest':'B2B' if h_rest==0 else f'{h_rest}d rest',
        'a_rest':'B2B' if a_rest==0 else f'{a_rest}d rest',
        'h_injured':h_injured,'a_injured':a_injured,
        'h_inj_adj':round(h_inj_adj*100,1),'a_inj_adj':round(a_inj_adj*100,1),
        'book_implied':book_imp,'model_ev':ml_ev,'odds_am':ml_am,'odds_dec':ml_dec,
        'ou_pick':ou_pick,'ou_confidence':ou_conf,'ou_line':ou_line,
        'ou_book_implied':ou_bimp,'ou_model_ev':ou_ev,'ou_odds_am':ou_am,
        'ou_conf_num':round(ou_conf*100,1),
    })

picks.sort(key=lambda x: -(x['model_ev'] or -999) if x['model_ev'] else -x['confidence'])

# ── Save picks ────────────────────────────────
prop_picks_today = []  # will be populated after build_props_html

if today not in record.get('by_day',{}):
    record.setdefault('by_day',{})[today] = {
        'W':0,'L':0,
        'picks':[{
            'home_abbr':p['home_abbr'],'away_abbr':p['away_abbr'],
            'pick_abbr':p['pick_abbr'],'conf':round(p['confidence']*100,1),
            'ou_pick':p['ou_pick'],'ou_line':p['ou_line'],
            'ou_conf':p['ou_conf_num'],
            'result':'','ou_result':'','home_score':0,'away_score':0,'total':0
        } for p in picks],
        'prop_picks': prop_picks_today,
    }
    save_record(record)
elif not record['by_day'][today].get('prop_picks'):
    record['by_day'][today]['prop_picks'] = prop_picks_today
    save_record(record)

# ── Save tonight's lines to historical_lines.csv ──────────────────────────────
if picks:
    lines_path = 'data/historical_lines.csv'
    new_lines  = []
    for p in picks:
        go = odds_map.get(p['home'], {})
        new_lines.append({
            'date':           today,
            'home_team':      p['home_abbr'],
            'away_team':      p['away_abbr'],
            'close_line':     p.get('ou_line', ''),
            'home_ml_dec':    go.get('home_odds_dec', ''),
            'away_ml_dec':    go.get('away_odds_dec', ''),
            'home_ml_am':     go.get('home_odds_am', ''),
            'away_ml_am':     go.get('away_odds_am', ''),
            'actual_total':   '',   # filled in by grade.py after game
            'home_score':     '',
            'away_score':     '',
        })
    new_df = pd.DataFrame(new_lines)
    if os.path.exists(lines_path):
        existing = pd.read_csv(lines_path)
        # Don't duplicate — skip dates already in file
        existing_dates = set(existing['date'].unique())
        new_df = new_df[~new_df['date'].isin(existing_dates)]
        if not new_df.empty:
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.to_csv(lines_path, index=False)
            print(f"  Lines saved: +{len(new_df)} rows → {lines_path}")
    else:
        new_df.to_csv(lines_path, index=False)
        print(f"  Lines saved: {len(new_df)} rows → {lines_path} (new file)")

# ── Stats ─────────────────────────────────────
at     = record.get('alltime',{"W":0,"L":0})
ou_at  = record.get('ou_alltime',{"W":0,"L":0})
tot    = at.get('W',0)+at.get('L',0)
ou_tot = ou_at.get('W',0)+ou_at.get('L',0)
pct    = at['W']/tot*100 if tot else 0
ou_pct = ou_at['W']/ou_tot*100 if ou_tot else 0
now    = datetime.now(timezone.utc)
this_month = now.strftime('%Y-%m')
month_rec  = record.get('by_month',{}).get(this_month,{"W":0,"L":0})
ou_month   = record.get('ou_by_month',{}).get(this_month,{"W":0,"L":0})

prop_at       = record.get('prop_alltime', {"W":0,"L":0})
prop_by_market = record.get('prop_by_market', {})
prop_month = record.get('prop_by_month', {}).get(this_month, {"W":0,"L":0})
prop_tot   = prop_at.get('W',0) + prop_at.get('L',0)
prop_pct   = prop_at['W']/prop_tot*100 if prop_tot else 0
# Playoff detection — used by prop model adjustment
_is_playoffs = (now.month >= 5) or (now.month == 4 and now.day >= 19)
yest_str   = (now-timedelta(days=1)).strftime('%Y-%m-%d')
yest_rec   = record.get('by_day',{}).get(yest_str,{"W":0,"L":0})

# ── Day tabs ──────────────────────────────────
day_tabs = ['<a href="picks.html" style="display:inline-block;padding:7px 14px;border-radius:20px;font-size:12px;font-weight:500;text-decoration:none;margin-right:6px;background:#111;color:#fff;">Today</a>']
for i in range(1,30):
    d    = (now-timedelta(days=i)).strftime('%Y-%m-%d')
    drec = record.get('by_day',{}).get(d,{})
    if not drec.get('picks'): continue
    dl = (now-timedelta(days=i)).strftime('%b %d')
    dw,dl2 = drec.get('W',0),drec.get('L',0)
    rs = f' {dw}-{dl2}' if (dw+dl2)>0 else ''
    rc = '#16a34a' if dw>dl2 else ('#dc2626' if dl2>dw else '#9ca3af')
    day_tabs.append(f'<a href="history/picks_{d}.html" style="display:inline-block;padding:7px 14px;border-radius:20px;font-size:12px;font-weight:500;text-decoration:none;margin-right:6px;background:#f3f4f6;color:#374151;">{dl}<span style="font-size:11px;color:{rc};">{rs}</span></a>')
    if len(day_tabs)>=15: break
tabs_html = ''.join(day_tabs)

# ── Record box helper ─────────────────────────
def conf_bands(prefix=''):
    if prefix == 'prop_':
        key = 'prop_by_conf'
    elif prefix == 'ou_':
        key = 'ou_by_conf'
    else:
        key = 'by_conf'
    rows = ''
    for band,label in [('50','50-55%'),('55','55-60%'),('60','60-65%'),('65','65-70%'),('70','70%+')]:
        b = record.get(key,{}).get(band,{"W":0,"L":0})
        t = b['W']+b['L']
        p = b['W']/t*100 if t else 0
        col = '#16a34a' if p>=55 else ('#dc2626' if t>0 else '#9ca3af')
        rows += (f'<div style="background:#fff;border-radius:8px;padding:10px 6px;text-align:center;border:0.5px solid #e5e7eb;">'
                 f'<div style="font-size:10px;color:#9ca3af;margin-bottom:3px;">{label}</div>'
                 f'<div style="font-size:13px;font-weight:500;">{b["W"]}-{b["L"]}</div>'
                 f'<div style="font-size:10px;color:{col};margin-top:1px;">{p:.0f}%</div></div>')
    return rows

def record_box(title, at_rec, month_rec, yest_rec, today_n, pct_val, bands_html):
    tot = at_rec.get('W',0)+at_rec.get('L',0)
    p   = at_rec['W']/tot*100 if tot else 0
    return f'''
<div style="background:#f9fafb;border-radius:12px;border:0.5px solid #e5e7eb;padding:16px 18px;margin-bottom:20px;">
  <div style="font-size:10px;font-weight:500;text-transform:uppercase;letter-spacing:.08em;color:#9ca3af;margin-bottom:12px;">{title}</div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:14px;">
    <div style="text-align:center;"><div style="font-size:10px;color:#9ca3af;margin-bottom:3px;">All time</div>
      <div style="font-size:22px;font-weight:500;color:#111;">{at_rec.get("W",0)}-{at_rec.get("L",0)}</div>
      <div style="font-size:11px;color:{"#16a34a" if p>=55 else "#dc2626"};">{p:.1f}%</div></div>
    <div style="text-align:center;"><div style="font-size:10px;color:#9ca3af;margin-bottom:3px;">This month</div>
      <div style="font-size:22px;font-weight:500;color:#111;">{month_rec.get("W",0)}-{month_rec.get("L",0)}</div></div>
    <div style="text-align:center;"><div style="font-size:10px;color:#9ca3af;margin-bottom:3px;">Yesterday</div>
      <div style="font-size:22px;font-weight:500;color:#111;">{yest_rec.get("W",0)}-{yest_rec.get("L",0)}</div></div>
    <div style="text-align:center;"><div style="font-size:10px;color:#9ca3af;margin-bottom:3px;">Today</div>
      <div style="font-size:22px;font-weight:500;color:#111;">{today_n}</div></div>
  </div>
  <div style="height:5px;background:#e5e7eb;border-radius:99px;overflow:hidden;margin-bottom:4px;">
    <div style="width:{min(p,100):.1f}%;height:100%;background:#16a34a;border-radius:99px;"></div></div>
  <div style="display:flex;justify-content:space-between;font-size:10px;color:#9ca3af;margin-bottom:14px;">
    <span>Hit rate</span><span>Break-even ~52%</span></div>
  <div style="font-size:10px;font-weight:500;text-transform:uppercase;letter-spacing:.07em;color:#9ca3af;margin-bottom:8px;">Record by confidence</div>
  <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:6px;">{bands_html}</div>
</div>'''

def injury_html(injured, adj):
    if not injured: return ''
    items = ''.join(
        f'<div style="font-size:11px;color:#dc2626;padding:2px 0;border-bottom:0.5px solid #f9fafb;">'
        f'{x["name"]} — {x["type"]} (OUT){"  gs/gm "+str(x["gs_pg"]) if x["gs_pg"]>0 else ""}</div>'
        for x in injured[:4])
    adj_str = f'<div style="font-size:10px;color:#dc2626;margin-top:3px;">Prob adj: {adj:+.1f}%</div>' if adj!=0 else ''
    return (f'<div style="margin-top:8px;"><div style="font-size:10px;color:#9ca3af;font-weight:500;'
            f'text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px;">Injuries</div>'
            f'{items}{adj_str}</div>')

# ── Props processing ───────────────────────────
_full_to_abbr = {v: k for k, v in abbr_to_full.items()}
def get_model_prob_for_prop(prop):
    market  = prop['market']
    player  = prop['player']
    line    = prop.get('line', 0)

    # ── Playoff adjustment factors ────────────────────────────────────────────
    # Playoff hockey has tighter defense, lower scoring, less ice time for role players
    # Based on historical regular season → playoff rate drops:
    #   Shots on goal: ~15% fewer (defensive systems tighten)
    #   Points:        ~12% fewer (less PP, tighter D)
    #   Saves:         ~8% fewer (fewer shots against in tight games)
    # Only apply during playoffs (is_playoffs set in build_html)
    is_po = _is_playoffs  # set earlier in build_html.py
    PLAYOFF_ADJ = {
        'player_shots_on_goal':   0.85,  # 15% fewer shots
        'player_points':          0.88,  # 12% fewer points
        'player_points_assists':  0.88,
        'player_goalie_saves':    0.92,  # 8% fewer saves
    }
    adj = PLAYOFF_ADJ.get(market, 1.0) if is_po else 1.0

    if market in ('player_points', 'player_points_assists'):
        match = fuzzy_match_player(player, skaters_prop_index)
        if match is None or skaters_prop.empty: return None, None
        row = skaters_prop[skaters_prop['name'] == match]
        if row.empty: return None, None
        ptspg = float(row['ptspg'].iloc[0]) * adj
        prob = 1 - poisson.cdf(int(line), ptspg)
        label = f"{ptspg:.2f} pts/gm" + (' (playoff adj)' if is_po else '')
        return round(prob, 4), label

    elif market == 'player_shots_on_goal':
        match = fuzzy_match_player(player, skaters_prop_index)
        if match is None or skaters_prop.empty: return None, None
        row = skaters_prop[skaters_prop['name'] == match]
        if row.empty: return None, None
        shotspg = float(row['shotspg'].iloc[0]) * adj
        # Opponent defense adjustment
        opp_abbr = prop.get('opp_abbr', '')
        opp_soga = _opp_soga.get(opp_abbr, _league_avg_soga)
        opp_adj = opp_soga / _league_avg_soga if _league_avg_soga > 0 else 1.0
        shotspg = shotspg * opp_adj
        prob = 1 - poisson.cdf(int(line), shotspg)
        label = f"{shotspg:.2f} sog/gm (playoff adj)" if is_po else f"{shotspg:.2f} sog/gm"
        return round(prob, 4), label

    elif market == 'player_goalie_saves':
        match = fuzzy_match_player(player, goalies_prop_index)
        if match is None or goalies_prop.empty: return None, None
        row = goalies_prop[goalies_prop['name'] == match]
        if row.empty: return None, None
        savespg = float(row['savespg'].iloc[0]) * adj
        prob = 1 - poisson.cdf(int(line), savespg)
        label = f"{savespg:.1f} saves/gm" + (' (playoff adj)' if is_po else '')
        return round(prob, 4), label

    return None, None

def build_props_html(props_map):
    """Build the full props section HTML."""
    if not props_map:
        return '<div style="text-align:center;padding:40px 0;color:#9ca3af;">No props available today.</div>', []

    MARKET_ORDER = ['player_shots_on_goal', 'player_goalie_saves', 'player_points', 'player_points_assists']
    MARKET_LABELS = {
        'player_shots_on_goal':  'Shots on Goal',
        'player_goalie_saves':   'Goalie Saves',
        'player_points':         'Points',
        'player_points_assists': 'Points + Assists',
    }

    # Collect all props with model edge
    all_edges = []
    for home_team, game_data in props_map.items():
        away_team = game_data['away']
        for prop in game_data['props']:
            # Pass opponent abbr for defense adjustment
            prop['opp_abbr'] = _full_to_abbr.get(away_team, '')
            model_prob, stat_label = get_model_prob_for_prop(prop)
            if model_prob is None:
                continue

            if prop['type'] == 'binary':
                book_imp  = prop['over_imp']
                dec_odds  = prop['over_dec']
                am_odds   = prop['over_am']
                edge      = model_prob - book_imp
                ev        = get_ev(model_prob, dec_odds)
                pick_str  = 'Yes'
            else:
                # Choose better side
                over_prob  = model_prob
                under_prob = 1 - model_prob
                book_over  = prop.get('over_imp', 0.5)
                book_under = prop.get('under_imp', 0.5)
                over_edge  = over_prob - book_over
                under_edge = under_prob - book_under
                if over_edge >= under_edge:
                    edge     = over_edge
                    ev       = get_ev(over_prob, prop.get('over_dec', 1.9))
                    am_odds  = prop.get('over_am', 'N/A')
                    pick_str = f"Over {prop['line']}"
                    model_prob_display = over_prob
                else:
                    edge     = under_edge
                    ev       = get_ev(under_prob, prop.get('under_dec', 1.9))
                    am_odds  = prop.get('under_am', 'N/A')
                    pick_str = f"Under {prop['line']}"
                    model_prob_display = under_prob
                model_prob = model_prob_display
                book_imp   = book_over if 'Over' in pick_str else book_under

            if ev is None or ev < -5:
                continue  # skip clear negative EV

            all_edges.append({
                'home':       home_team,
                'away':       away_team,
                'player':     prop['player'],
                'market':     prop['market'],
                'label':      MARKET_LABELS.get(prop['market'], prop['label']),
                'pick':       pick_str,
                'model_prob': model_prob,
                'book_imp':   book_imp,
                'edge':       edge,
                'ev':         ev,
                'am_odds':    am_odds,
                'stat_label': stat_label,
            })

    if not all_edges:
        return '<div style="text-align:center;padding:40px 0;color:#9ca3af;">No props with positive EV found today.</div>', []

    # Sort by EV descending
    all_edges.sort(key=lambda x: -x['ev'])

    # Group by market for display
    by_market = {}
    for e in all_edges:
        by_market.setdefault(e['market'], []).append(e)

    html_parts = []
    for market in MARKET_ORDER:
        if market not in by_market:
            continue
        entries = by_market[market][:15]  # cap at 15 per market
        label   = MARKET_LABELS.get(market, market)

        html_parts.append(
            f'<div style="font-size:11px;font-weight:600;text-transform:uppercase;'
            f'letter-spacing:.08em;color:#9ca3af;margin:18px 0 8px;">{label}</div>'
        )

        for e in entries:
            ev_col = '#16a34a' if e['ev'] >= 8 else ('#65a30d' if e['ev'] >= 3 else '#d97706')
            edge_pct = e['edge'] * 100
            html_parts.append(f'''
<div style="background:#fff;border-radius:10px;border:0.5px solid #e5e7eb;
            margin-bottom:8px;padding:12px 14px;">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;">
    <div style="flex:1;min-width:0;">
      <div style="font-size:14px;font-weight:500;color:#111;
                  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{e["player"]}</div>
      <div style="font-size:11px;color:#9ca3af;margin-top:2px;">
        {e["away"]} @ {e["home"]}
        {f'· {e["stat_label"]}' if e["stat_label"] else ''}
      </div>
    </div>
    <div style="text-align:right;flex-shrink:0;margin-left:10px;">
      <div style="font-size:14px;font-weight:600;color:#111;">{e["pick"]}</div>
      <div style="font-size:12px;color:#9ca3af;">{e["am_odds"]}</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:10px;margin-top:8px;
              padding:6px 10px;background:#f9fafb;border-radius:6px;">
    <div>
      <div style="font-size:9px;color:#9ca3af;">BOOK</div>
      <div style="font-size:13px;font-weight:500;">{e["book_imp"]:.0%}</div>
    </div>
    <div>
      <div style="font-size:9px;color:#9ca3af;">MODEL</div>
      <div style="font-size:13px;font-weight:500;">{e["model_prob"]:.0%}</div>
    </div>
    <div>
      <div style="font-size:9px;color:#9ca3af;">EDGE</div>
      <div style="font-size:13px;font-weight:500;color:{"#16a34a" if edge_pct>0 else "#dc2626"};">{edge_pct:+.1f}%</div>
    </div>
    <div style="margin-left:auto;">
      <div style="font-size:9px;color:#9ca3af;">EV/$100</div>
      <div style="font-size:15px;font-weight:600;color:{ev_col};">{e["ev"]:+.1f}</div>
    </div>
  </div>
</div>''')

    # Build prop picks list for saving to record
    prop_picks_today = []
    for e in all_edges:
        conf_pct = round(e['model_prob'] * 100, 1)
        home_abbr = _full_to_abbr.get(e['home'], e['home'])
        away_abbr = _full_to_abbr.get(e['away'], e['away'])
        prop_picks_today.append({
            'home_abbr': home_abbr,
            'away_abbr': away_abbr,
            'player':    e['player'],
            'market':    e['market'],
            'pick':      e['pick'],
            'line':      float(e['pick'].split()[-1]) if e['pick'].split()[-1].replace('.','').isdigit() else e.get('line', 0),
            'conf':      conf_pct,
            'ev':        round(e['ev'], 1),
            'result':    '',
            'actual':    None,
        })

    return ''.join(html_parts), prop_picks_today




# ── ML Cards ──────────────────────────────────
def ml_card(p, idx):
    cn       = p['confidence']*100
    pih      = p['winner']==p['home']
    hp       = p['confidence'] if pih else 1-p['confidence']
    ap       = 1-hp
    hgs      = f"{p['h_gsax']:+.1f}" if p['h_goalie']!='TBD' else 'N/A'
    ags      = f"{p['a_gsax']:+.1f}" if p['a_goalie']!='TBD' else 'N/A'
    hf       = ' ⚠️' if p['h_inj_adj']<-1.5 else ''
    af       = ' ⚠️' if p['a_inj_adj']<-1.5 else ''
    cc       = '#15803d' if cn>=65 else ('#16a34a' if cn>=60 else ('#d97706' if cn>=55 else '#374151'))
    bc       = '#16a34a' if pih else '#3b82f6'

    odds_row = ''
    if p.get('odds_am'):
        ev    = p.get('model_ev',0) or 0
        evc   = '#16a34a' if ev>5 else ('#d97706' if ev>0 else '#dc2626')
        evl   = f'+{ev:.1f}' if ev>0 else f'{ev:.1f}'
        imp   = p.get('book_implied',0) or 0
        odds_row = f'''
        <div style="display:flex;align-items:center;gap:10px;margin-top:6px;padding:6px 10px;background:#f9fafb;border-radius:6px;">
          <div><div style="font-size:10px;color:#9ca3af;">ODDS</div><div style="font-size:14px;font-weight:500;color:#111;">{p["odds_am"]}</div></div>
          <div><div style="font-size:10px;color:#9ca3af;">BOOK</div><div style="font-size:14px;font-weight:500;color:#111;">{imp:.0%}</div></div>
          <div><div style="font-size:10px;color:#9ca3af;">MODEL</div><div style="font-size:14px;font-weight:500;color:#111;">{p["confidence"]:.0%}</div></div>
          <div style="margin-left:auto;"><div style="font-size:10px;color:#9ca3af;">EV/$100</div><div style="font-size:15px;font-weight:600;color:{evc};">{evl}</div></div>
        </div>'''

    h_inj = injury_html(p['h_injured'],p['h_inj_adj'])
    a_inj = injury_html(p['a_injured'],p['a_inj_adj'])

    return f'''
<div style="background:#fff;border-radius:12px;border:0.5px solid #e5e7eb;margin-bottom:12px;overflow:hidden;">
  <div onclick="toggle({idx})" style="padding:16px 18px;cursor:pointer;user-select:none;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
      <span style="font-size:11px;color:#9ca3af;">{p['imp_label']}</span>
      <span style="font-size:13px;font-weight:500;color:{cc};">{p["confidence"]:.0%} confidence</span>
    </div>
    <div style="display:flex;align-items:center;justify-content:space-between;gap:6px;">
      <div style="flex:1;min-width:0;">
        <div style="font-size:15px;font-weight:500;color:#111;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{p['home']}{hf}</div>
        <div style="font-size:11px;color:#9ca3af;">L10: {p['h_l10w']}-{p['h_l10l']} · {p['h_rest']} · Home</div>
        <div style="font-size:11px;color:#9ca3af;margin-top:1px;">{p['h_goalie']} <span style="color:{'#16a34a' if p['h_gsax']>=0 else '#dc2626'};">GSAx {hgs}</span></div>
      </div>
      <div style="text-align:center;padding:0 10px;flex-shrink:0;">
        <div style="font-size:10px;color:#9ca3af;margin-bottom:3px;">VS</div>
        <div style="font-size:22px;font-weight:500;color:#111;">–</div>
      </div>
      <div style="flex:1;min-width:0;text-align:right;">
        <div style="font-size:15px;font-weight:500;color:#111;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{p['away']}{af}</div>
        <div style="font-size:11px;color:#9ca3af;">L10: {p['a_l10w']}-{p['a_l10l']} · {p['a_rest']} · Away</div>
        <div style="font-size:11px;color:#9ca3af;margin-top:1px;">{p['a_goalie']} <span style="color:{'#16a34a' if p['a_gsax']>=0 else '#dc2626'};">GSAx {ags}</span></div>
      </div>
    </div>
    <div style="margin-top:10px;">
      <div style="height:5px;background:#f3f4f6;border-radius:99px;overflow:hidden;display:flex;">
        <div style="width:{hp*100:.0f}%;background:{""+bc if pih else "#d1d5db"};border-radius:99px 0 0 99px;"></div>
        <div style="width:{ap*100:.0f}%;background:{""+bc if not pih else "#d1d5db"};border-radius:0 99px 99px 0;"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:10px;color:#9ca3af;margin-top:3px;">
        <span>{p['home'].split()[-1]} {hp*100:.0f}%</span>
        <span style="color:#374151;font-weight:500;">Pick: {p['winner']}</span>
        <span>{p['away'].split()[-1]} {ap*100:.0f}%</span>
      </div>
    </div>
    {odds_row}
    <div style="margin-top:8px;border-top:0.5px solid #f3f4f6;padding-top:8px;">
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div><div style="font-size:10px;color:#9ca3af;margin-bottom:2px;">MODEL PICK</div>
        <div style="font-size:15px;font-weight:600;color:#111;">{p['winner']}</div></div>
        <div style="font-size:11px;color:#d1d5db;" id="hint{idx}">tap for details ▾</div>
      </div>
    </div>
  </div>
  <div id="body{idx}" style="display:none;border-top:0.5px solid #f3f4f6;padding:16px 18px;">
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px;">
      <div style="background:#f9fafb;border-radius:8px;padding:10px;">
        <div style="font-size:10px;color:#9ca3af;margin-bottom:4px;font-weight:500;text-transform:uppercase;">HOME — {p['home_abbr']}</div>
        <div style="font-size:12px;color:#374151;">🥅 {p['h_goalie']}</div>
        <div style="font-size:12px;color:{'#16a34a' if p['h_gsax']>=0 else '#dc2626'};">GSAx: {hgs}</div>
        <div style="font-size:12px;color:#374151;">Last 10: {p['h_l10w']}-{p['h_l10l']}</div>
        <div style="font-size:12px;color:#374151;">Rest: {p['h_rest']}</div>
        {h_inj}
      </div>
      <div style="background:#f9fafb;border-radius:8px;padding:10px;">
        <div style="font-size:10px;color:#9ca3af;margin-bottom:4px;font-weight:500;text-transform:uppercase;">AWAY — {p['away_abbr']}</div>
        <div style="font-size:12px;color:#374151;">🥅 {p['a_goalie']}</div>
        <div style="font-size:12px;color:{'#16a34a' if p['a_gsax']>=0 else '#dc2626'};">GSAx: {ags}</div>
        <div style="font-size:12px;color:#374151;">Last 10: {p['a_l10w']}-{p['a_l10l']}</div>
        <div style="font-size:12px;color:#374151;">Rest: {p['a_rest']}</div>
        {a_inj}
      </div>
    </div>
    <div style="text-align:center;margin-top:8px;">
      <span onclick="toggle({idx})" style="font-size:11px;color:#d1d5db;cursor:pointer;">collapse ▴</span>
    </div>
  </div>
</div>'''

# ── O/U Cards ─────────────────────────────────
def ou_card(p, idx):
    oc  = p['ou_confidence']
    ocn = oc*100
    occ = '#15803d' if ocn>=65 else ('#16a34a' if ocn>=60 else ('#d97706' if ocn>=55 else '#9ca3af'))
    skip = '<span style="font-size:10px;color:#9ca3af;margin-left:6px;">· low conf</span>' if ocn<60 else ''
    ou_label = f"{p['ou_pick']} {p['ou_line']}"

    odds_row = ''
    if p.get('ou_odds_am'):
        ev    = p.get('ou_model_ev',0) or 0
        evc   = '#16a34a' if ev>5 else ('#d97706' if ev>0 else '#dc2626')
        evl   = f'+{ev:.1f}' if ev>0 else f'{ev:.1f}'
        imp   = p.get('ou_book_implied',0) or 0
        odds_row = f'''
        <div style="display:flex;align-items:center;gap:10px;margin-top:6px;padding:6px 10px;background:#f9fafb;border-radius:6px;">
          <div><div style="font-size:10px;color:#9ca3af;">ODDS</div><div style="font-size:14px;font-weight:500;color:#111;">{p["ou_odds_am"]}</div></div>
          <div><div style="font-size:10px;color:#9ca3af;">BOOK</div><div style="font-size:14px;font-weight:500;color:#111;">{imp:.0%}</div></div>
          <div><div style="font-size:10px;color:#9ca3af;">MODEL</div><div style="font-size:14px;font-weight:500;color:#111;">{oc:.0%}</div></div>
          <div style="margin-left:auto;"><div style="font-size:10px;color:#9ca3af;">EV/$100</div><div style="font-size:15px;font-weight:600;color:{evc};">{evl}</div></div>
        </div>'''

    return f'''
<div style="background:#fff;border-radius:12px;border:0.5px solid #e5e7eb;margin-bottom:12px;padding:14px 18px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
    <span style="font-size:11px;color:#9ca3af;">{p['imp_label']}</span>
    <span style="font-size:13px;font-weight:500;color:{occ};">{oc:.0%} confidence{skip}</span>
  </div>
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
    <div style="font-size:15px;font-weight:500;color:#111;">{p['home']}</div>
    <div style="font-size:13px;color:#9ca3af;">vs</div>
    <div style="font-size:15px;font-weight:500;color:#111;">{p['away']}</div>
  </div>
  <div style="display:flex;align-items:center;gap:8px;padding:8px 10px;background:#f9fafb;border-radius:6px;border-left:3px solid {occ};">
    <div style="font-size:10px;color:#9ca3af;font-weight:500;text-transform:uppercase;">O/U PICK</div>
    <div style="font-size:16px;font-weight:700;color:{occ};">{ou_label}</div>
    <div style="font-size:12px;color:{occ};">{oc:.0%}</div>
  </div>
  {odds_row}
  <div style="margin-top:8px;display:flex;gap:16px;font-size:11px;color:#9ca3af;">
    <span>🥅 {p['h_goalie']} <span style="color:{'#16a34a' if p['h_gsax']>=0 else '#dc2626'};">GSAx {p['h_gsax']:+.1f}</span></span>
    <span>🥅 {p['a_goalie']} <span style="color:{'#16a34a' if p['a_gsax']>=0 else '#dc2626'};">GSAx {p['a_gsax']:+.1f}</span></span>
  </div>
</div>'''

props_html, prop_picks_today = build_props_html(props_map)

# Save prop_picks to record now that we have them
if prop_picks_today and not record.get('by_day',{}).get(today,{}).get('prop_picks'):
    record.setdefault('by_day',{}).setdefault(today,{})['prop_picks'] = prop_picks_today
    save_record(record)

ml_cards  = '\n'.join(ml_card(p,i)    for i,p in enumerate(picks))
ou_cards  = '\n'.join(ou_card(p,i+100) for i,p in enumerate(
    sorted(picks, key=lambda x: -x['ou_confidence'])))

# ── History pages ─────────────────────────────
def build_history_page(date_str):
    import json as _json
    drec      = record.get('by_day',{}).get(date_str,{})
    dpicks    = drec.get('picks',[])
    dprops    = drec.get('prop_picks',[])
    dlabel    = datetime.strptime(date_str,'%Y-%m-%d').strftime('%A, %B %d %Y')

    # Day nav
    ht = ['<a href="../picks.html" style="display:inline-block;padding:7px 14px;border-radius:20px;font-size:12px;font-weight:500;text-decoration:none;margin-right:6px;background:#f3f4f6;color:#374151;">Today</a>']
    for i in range(1,30):
        d   = (datetime.now(timezone.utc)-timedelta(days=i)).strftime('%Y-%m-%d')
        dr  = record.get('by_day',{}).get(d,{})
        if not dr.get('picks'): continue
        dl  = (datetime.now(timezone.utc)-timedelta(days=i)).strftime('%b %d')
        dw2,dl2 = dr.get('W',0),dr.get('L',0)
        rs  = f' {dw2}-{dl2}' if (dw2+dl2)>0 else ''
        rc  = '#16a34a' if dw2>dl2 else ('#dc2626' if dl2>dw2 else '#9ca3af')
        active = d == date_str
        ht.append(f'<a href="picks_{d}.html" style="display:inline-block;padding:7px 14px;border-radius:20px;font-size:12px;font-weight:500;text-decoration:none;margin-right:6px;background:{"#111" if active else "#f3f4f6"};color:{"#fff" if active else "#374151"};">{dl}<span style="font-size:11px;color:{rc};">{rs}</span></a>')
        if len(ht)>=8: break

    # Summary
    dw   = drec.get('W',0); dl2 = drec.get('L',0)
    ou_w = sum(1 for p in dpicks if p.get('ou_result')=='W')
    ou_l = sum(1 for p in dpicks if p.get('ou_result')=='L')
    pr_w = sum(1 for p in dprops if p.get('result')=='W')
    pr_l = sum(1 for p in dprops if p.get('result')=='L')

    def summary_box(label, w, l):
        col = '#16a34a' if w>l else ('#dc2626' if l>w else '#9ca3af')
        txt = '✓ Win' if w>l else ('✗ Loss' if l>w else 'Even')
        return (f'<div style="background:#fff;border-radius:12px;border:0.5px solid #e5e7eb;padding:14px;text-align:center;">'
                f'<div style="font-size:11px;color:#9ca3af;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;">{label}</div>'
                f'<div style="font-size:26px;font-weight:500;">{w}-{l}</div>'
                f'<div style="font-size:12px;color:{col};margin-top:2px;">{txt}</div></div>')

    summary_html = (f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:20px;">'
                    f'{summary_box("Moneyline", dw, dl2)}'
                    f'{summary_box("Over/Under", ou_w, ou_l)}'
                    f'{summary_box("Props", pr_w, pr_l)}'
                    f'</div>')

    # Build pick data for JS
    ml_data   = []
    for p in dpicks:
        ml_data.append({
            'away': p.get('away_abbr',''), 'home': p.get('home_abbr',''),
            'pick': p.get('pick_abbr',''), 'conf': p.get('conf',0),
            'result': p.get('result',''),
            'score': f"{p.get('away_score',0)}-{p.get('home_score',0)}" if p.get('result') else '–',
            'ou_pick': p.get('ou_pick',''), 'ou_line': p.get('ou_line',''),
            'ou_conf': p.get('ou_conf',0), 'ou_result': p.get('ou_result',''),
            'total': p.get('total','–'),
        })

    prop_data = []
    for p in dprops:
        prop_data.append({
            'player': p.get('player',''), 'pick': p.get('pick',''),
            'conf': p.get('conf',0), 'ev': p.get('ev',0),
            'result': p.get('result',''), 'actual': p.get('actual','–'),
            'market': p.get('market',''),
        })

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>NHL Picks · {date_str}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f3f4f6;color:#111;min-height:100vh}}
.wrap{{max-width:680px;margin:0 auto;padding:20px 14px 60px}}
.card{{background:#fff;border-radius:12px;border:0.5px solid #e5e7eb;margin-bottom:10px;padding:14px 18px;}}
.badge{{display:inline-block;padding:2px 8px;border-radius:6px;font-size:12px;font-weight:600;}}
.win{{background:#dcfce7;color:#16a34a;}} .loss{{background:#fee2e2;color:#dc2626;}} .pending{{background:#f3f4f6;color:#9ca3af;}}
.tab-btn{{flex:1;padding:8px;border-radius:8px;border:0.5px solid #e5e7eb;font-size:13px;font-weight:500;cursor:pointer;}}
.sort-btn{{padding:6px 12px;border-radius:8px;border:0.5px solid #e5e7eb;background:#fff;font-size:12px;cursor:pointer;color:#374151;}}
.sort-btn.active{{background:#111;color:#fff;border-color:#111;}}
</style>
</head>
<body>
<div class="wrap">
  <div style="margin-bottom:16px;">
    <h1 style="font-size:22px;font-weight:500;">NHL Picks</h1>
    <div style="font-size:13px;color:#9ca3af;margin-top:3px;">{dlabel}</div>
  </div>
  <div style="overflow-x:auto;white-space:nowrap;margin-bottom:16px;padding-bottom:4px;">{"".join(ht)}</div>
  {summary_html}

  <!-- Tabs -->
  <div style="display:flex;gap:6px;margin-bottom:16px;">
    <button class="tab-btn" id="tab-ml" onclick="showTab('ml')" style="background:#111;color:#fff;">Moneyline</button>
    <button class="tab-btn" id="tab-ou" onclick="showTab('ou')" style="background:#f3f4f6;color:#374151;">Over/Under</button>
    <button class="tab-btn" id="tab-props" onclick="showTab('props')" style="background:#f3f4f6;color:#374151;">Props</button>
  </div>

  <div id="view-ml"></div>
  <div id="view-ou" style="display:none;"></div>
  <div id="view-props" style="display:none;"><div id="props-record-box"></div>
  PROPS_RECORD_PLACEHOLDER
</div>
</div>

<script>
var ML_DATA    = {_json.dumps(ml_data)};
var PROPS_RECORD_HTML = {_json.dumps(record_box('Props Record', prop_at, prop_month, {"W":0,"L":0}, 0, prop_pct, conf_bands('prop_')))};
var PROP_DATA  = {_json.dumps(prop_data)};
var currentTab = 'ml';

function badge(r){{
  if(r==='W') return '<span class="badge win">W</span>';
  if(r==='L') return '<span class="badge loss">L</span>';
  return '<span class="badge pending">–</span>';
}}

function showTab(t){{
  if(t==='props'){{
    var rec = document.getElementById('props-record-box');
    if(rec && PROPS_RECORD_HTML) rec.innerHTML = PROPS_RECORD_HTML;
  }}
  currentTab = t;
  ['ml','ou','props'].forEach(function(x){{
    document.getElementById('view-'+x).style.display = x===t?'block':'none';
    var b = document.getElementById('tab-'+x);
    b.style.background = x===t?'#111':'#f3f4f6';
    b.style.color      = x===t?'#fff':'#374151';
  }});
}}

function renderML(){{
  var html = '';
  ML_DATA.forEach(function(p){{
    html += '<div class="card"><div style="display:flex;justify-content:space-between;align-items:flex-start;">'
      + '<div><div style="font-size:13px;color:#9ca3af;margin-bottom:2px;">'+p.away+' @ '+p.home+'</div>'
      + '<div style="font-size:16px;font-weight:500;">Pick: '+p.pick+'</div>'
      + '<div style="font-size:12px;color:#9ca3af;margin-top:3px;">'+p.conf.toFixed(0)+'% confidence</div></div>'
      + '<div style="text-align:right;">'+badge(p.result)
      + '<div style="font-size:12px;color:#9ca3af;margin-top:6px;">'+p.score+'</div></div>'
      + '</div></div>';
  }});
  document.getElementById('view-ml').innerHTML = html || '<div style="color:#9ca3af;padding:20px 0;">No picks.</div>';
}}

function renderOU(){{
  var html = '';
  ML_DATA.forEach(function(p){{
    html += '<div class="card"><div style="display:flex;justify-content:space-between;align-items:flex-start;">'
      + '<div><div style="font-size:13px;color:#9ca3af;margin-bottom:2px;">'+p.away+' @ '+p.home+'</div>'
      + '<div style="font-size:16px;font-weight:500;">'+p.ou_pick+' '+p.ou_line+'</div>'
      + '<div style="font-size:12px;color:#9ca3af;margin-top:3px;">'+p.ou_conf.toFixed(0)+'% confidence</div></div>'
      + '<div style="text-align:right;">'+badge(p.ou_result)
      + '<div style="font-size:12px;color:#9ca3af;margin-top:6px;">Total: '+p.total+'</div></div>'
      + '</div></div>';
  }});
  document.getElementById('view-ou').innerHTML = html || '<div style="color:#9ca3af;padding:20px 0;">No picks.</div>';
}}

function renderProps(){{
  if(!PROP_DATA.length){{
    document.getElementById('view-props').innerHTML = '<div style="color:#9ca3af;padding:20px 0;">No props saved for this day.</div>';
    return;
  }}
  var sorted = PROP_DATA.slice().sort(function(a,b){{return b.conf-a.conf;}});
  var html = '';
  sorted.forEach(function(p){{
    var confCol = p.conf>=65?'#15803d':p.conf>=60?'#16a34a':p.conf>=55?'#d97706':'#374151';
    var mkt = p.market.replace('player_','').replace(/_/g,' ');
    html += '<div class="card"><div style="display:flex;justify-content:space-between;align-items:flex-start;">'
      + '<div><div style="font-size:14px;font-weight:500;">'+p.player+'</div>'
      + '<div style="font-size:11px;color:#9ca3af;margin-top:2px;">'+mkt+'</div></div>'
      + '<div style="text-align:right;"><div style="font-size:14px;font-weight:600;">'+p.pick+'</div>'
      + '<div style="font-size:11px;font-weight:500;color:'+confCol+';">'+p.conf.toFixed(0)+'% conf</div>'
      + badge(p.result)
      + (p.actual!==undefined&&p.actual!=='–'?'<div style="font-size:11px;color:#9ca3af;">actual: '+p.actual+'</div>':'')
      + '</div></div></div>';
  }});
  document.getElementById('view-props').innerHTML = html;
}}

renderML(); renderOU(); renderProps();
</script>
</body></html>'''

import os as _os
_os.makedirs('history', exist_ok=True)
for i in range(1,30):
    d = (now-timedelta(days=i)).strftime('%Y-%m-%d')
    if record.get('by_day',{}).get(d,{}).get('picks'):
        with open(f'history/picks_{d}.html','w',encoding='utf-8') as f:
            f.write(build_history_page(d))

# ── Yest O/U record ───────────────────────────
yest_ou = {"W":0,"L":0}
yd = record.get('by_day',{}).get(yest_str,{})
for p in yd.get('picks',[]):
    r = p.get('ou_result','')
    if r in ('W','L'): yest_ou[r] = yest_ou.get(r,0)+1

# ── Main HTML ─────────────────────────────────
prop_record_box = record_box('Props Record', prop_at, prop_month, {"W":0,"L":0}, 0, prop_pct, conf_bands('prop_'))
html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>NHL Picks</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f3f4f6;color:#111;min-height:100vh}}
  .wrap{{max-width:680px;margin:0 auto;padding:20px 14px 60px}}
  a{{cursor:pointer}}
  @media(max-width:480px){{.wrap{{padding:14px 10px 40px}}}}
</style>
</head>
<body>
<div class="wrap">

  <div style="margin-bottom:20px;">
    <h1 style="font-size:22px;font-weight:500;">NHL Picks</h1>
    <div style="font-size:13px;color:#9ca3af;margin-top:3px;">{today_display} · {len(picks)} games</div>
  </div>

  <div style="overflow-x:auto;white-space:nowrap;margin-bottom:16px;padding-bottom:4px;">{tabs_html}</div>

  <!-- VIEW TABS -->
  <div style="display:flex;gap:6px;margin-bottom:20px;">
    <button onclick="showView('ml')" id="tab-ml"
      style="flex:1;padding:8px;border-radius:8px;border:0.5px solid #e5e7eb;
             background:#111;color:#fff;font-size:13px;font-weight:500;cursor:pointer;">
      Moneyline
    </button>
    <button onclick="showView('ou')" id="tab-ou"
      style="flex:1;padding:8px;border-radius:8px;border:0.5px solid #e5e7eb;
             background:#f3f4f6;color:#374151;font-size:13px;font-weight:500;cursor:pointer;">
      Over / Under
    </button>
    <button onclick="showView('props')" id="tab-props"
      style="flex:1;padding:8px;border-radius:8px;border:0.5px solid #e5e7eb;
             background:#f3f4f6;color:#374151;font-size:13px;font-weight:500;cursor:pointer;">
      Props
    </button>
  </div>

  <!-- MONEYLINE SECTION -->
  <div id="view-ml">
  <div style="font-size:13px;font-weight:600;color:#111;margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid #e5e7eb;">
    Moneyline Picks
  </div>

  {record_box('Moneyline Record', at, month_rec, yest_rec, len(picks), pct, conf_bands())}

  <div style="display:flex;justify-content:flex-end;margin-bottom:12px;font-size:11px;color:#9ca3af;gap:5px;">
    EV: <span style="color:#16a34a;">+8 excellent</span> · <span style="color:#65a30d;">+3 good</span> · <span style="color:#d97706;">marginal</span> · <span style="color:#dc2626;">neg = skip</span>
  </div>

  {ml_cards if picks else '<div style="text-align:center;padding:40px 0;color:#9ca3af;">No games today.</div>'}

  </div><!-- end view-ml -->

  <!-- O/U SECTION -->
  <div id="view-ou" style="display:none;">
  <div style="font-size:13px;font-weight:600;color:#111;margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid #e5e7eb;">
    Over / Under Picks
  </div>

  {record_box('O/U Record', ou_at, ou_month, yest_ou, len(picks), ou_pct, conf_bands('ou_'))}

  <div style="display:flex;justify-content:flex-end;margin-bottom:12px;font-size:11px;color:#9ca3af;">
    60%+ confidence recommended · grey = low confidence, use caution
  </div>

  {ou_cards if picks else '<div style="text-align:center;padding:40px 0;color:#9ca3af;">No games today.</div>'}

  </div><!-- end view-ou -->

  <!-- PROPS SECTION -->
  <div id="view-props" style="display:none;">
  PROPS_RECORD_PLACEHOLDER
  <div style="font-size:13px;font-weight:600;color:#111;margin-bottom:6px;padding-bottom:8px;border-bottom:1px solid #e5e7eb;">
    Player Props
  </div>
  <div style="font-size:11px;color:#9ca3af;margin-bottom:14px;">
    Model edge vs DraftKings implied probability · sorted by EV · +EV only shown
  </div>
  {props_html}
  </div><!-- end view-props -->

  <div style="margin-top:20px;padding:12px 14px;background:#fffbeb;border-radius:8px;border:0.5px solid #fde68a;">
    <p style="font-size:11px;color:#92400e;line-height:1.6;">For informational purposes only. Gamble responsibly.</p>
  </div>
</div>
<script>
function showView(v){{
  document.getElementById('view-ml').style.display    = v==='ml'    ? 'block' : 'none';
  document.getElementById('view-ou').style.display    = v==='ou'    ? 'block' : 'none';
  document.getElementById('view-props').style.display = v==='props' ? 'block' : 'none';
  ['ml','ou','props'].forEach(t=>{{
    var btn = document.getElementById('tab-'+t);
    btn.style.background = v===t ? '#111' : '#f3f4f6';
    btn.style.color      = v===t ? '#fff' : '#374151';
  }});
}}
function toggle(i){{
  var b=document.getElementById('body'+i);
  var h=document.getElementById('hint'+i);
  var o=b.style.display!=='none';
  b.style.display=o?'none':'block';
  h.innerHTML=o?'tap for details &#9662;':'collapse &#9652;';
}}
</script>
</body></html>'''

# Inject props record box into html
# Build per-market mini records
market_records_html = ''
MARKET_ORDER = [
    ('player_shots_on_goal', 'Shots on Goal'),
    ('player_points',        'Points'),
    ('player_points_assists','Points + Assists'),
    ('player_goalie_saves',  'Saves'),
]
for mkey, mlabel in MARKET_ORDER:
    mdata = prop_by_market.get(mkey)
    if not mdata:
        continue
    mat = mdata.get('alltime', {"W":0,"L":0})
    mt  = mat['W'] + mat['L']
    mp  = mat['W']/mt*100 if mt else 0
    mconf = mdata.get('by_conf', {})
    band_html = ''
    for band, blabel in [('60','60-65%'),('65','65-70%'),('70','70%+')]:
        b = mconf.get(band, {"W":0,"L":0})
        bt = b['W']+b['L']
        bp = b['W']/bt*100 if bt else 0
        bcol = '#16a34a' if bp>=55 else ('#dc2626' if bt>0 else '#9ca3af')
        band_html += (f'<div style="text-align:center;padding:6px 4px;">'
                      f'<div style="font-size:10px;color:#9ca3af;">{blabel}</div>'
                      f'<div style="font-size:12px;font-weight:500;">{b["W"]}-{b["L"]}</div>'
                      f'<div style="font-size:10px;color:{bcol};">{bp:.0f}%</div></div>')
    mcol = '#16a34a' if mp>=52 else '#dc2626'
    market_records_html += (
        f'<div style="background:#f9fafb;border-radius:10px;border:0.5px solid #e5e7eb;'
        f'padding:12px 14px;margin-bottom:10px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
        f'<div style="font-size:12px;font-weight:600;color:#111;">{mlabel}</div>'
        f'<div style="font-size:12px;font-weight:500;color:{mcol};">{mat["W"]}-{mat["L"]} ({mp:.0f}%)</div>'
        f'</div>'
        f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:4px;">{band_html}</div>'
        f'</div>'
    )

props_record_html = (
    record_box('Props Record', prop_at, prop_month, {"W":0,"L":0}, 0, prop_pct, conf_bands('prop_')) +
    ('<div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;'
     'color:#9ca3af;margin:16px 0 8px;">Record by Market</div>' + market_records_html
     if market_records_html else '') +
    '<div style="display:flex;justify-content:flex-end;margin-bottom:12px;font-size:11px;color:#9ca3af;">60%+ confidence recommended</div>'
)
html = html.replace('PROPS_RECORD_PLACEHOLDER', props_record_html)

with open('picks.html','w',encoding='utf-8') as f:
    f.write(html)

print(f"\nDone! Run: start picks.html")
print(f"ML:  {at.get('W',0)}-{at.get('L',0)} ({pct:.1f}%)")
print(f"O/U: {ou_at.get('W',0)}-{ou_at.get('L',0)} ({ou_pct:.1f}%)")
print(f"\nTonight:")
for p in picks:
    ml_ev = f"EV {p['model_ev']:+.1f}" if p['model_ev'] else ''
    ou_ev = f"EV {p['ou_model_ev']:+.1f}" if p.get('ou_model_ev') else ''
    print(f"  ML:  {p['winner']:<28} {p['confidence']:.0%} {ml_ev}")
    print(f"  O/U: {p['ou_pick']} {p['ou_line']:<25} {p['ou_confidence']:.0%} {ou_ev}")
    print()