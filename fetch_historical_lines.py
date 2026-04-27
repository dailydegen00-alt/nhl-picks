"""
fetch_historical_lines.py (v2)
SBR serves HTML tables with .xlsx extension — parse with pandas read_html.
"""
import requests
import pandas as pd
import numpy as np
import os

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

SBR_TO_ABBR = {
    'Anaheim':'ANA','Boston':'BOS','Buffalo':'BUF','Calgary':'CGY',
    'Carolina':'CAR','Chicago':'CHI','Colorado':'COL','Columbus':'CBJ',
    'Dallas':'DAL','Detroit':'DET','Edmonton':'EDM','Florida':'FLA',
    'Los Angeles':'LAK','LosAngeles':'LAK','Minnesota':'MIN',
    'Montreal':'MTL','Nashville':'NSH','New Jersey':'NJD','NJ':'NJD',
    'NY Rangers':'NYR','NYRangers':'NYR','NY Islanders':'NYI','NYIslanders':'NYI',
    'Ottawa':'OTT','Philadelphia':'PHI','Pittsburgh':'PIT',
    'San Jose':'SJS','SanJose':'SJS','Seattle':'SEA',
    'St. Louis':'STL','StLouis':'STL','Tampa Bay':'TBL','TampaBay':'TBL',
    'Toronto':'TOR','Utah':'UTA','Vancouver':'VAN','Vegas':'VGK',
    'Washington':'WSH','Winnipeg':'WPG','Arizona':'ARI','Phoenix':'ARI',
}

def norm(name):
    name = str(name).strip()
    if name in SBR_TO_ABBR: return SBR_TO_ABBR[name]
    nospace = name.replace(' ','')
    if nospace in SBR_TO_ABBR: return SBR_TO_ABBR[nospace]
    return name[:3].upper()

SEASONS = {
    '2023-24': ('https://www.sportsbookreviewsonline.com/scoresoddsarchives/nhl/nhl%20odds%202023-24.xlsx', 2023),
    '2024-25': ('https://www.sportsbookreviewsonline.com/scoresoddsarchives/nhl/nhl%20odds%202024-25.xlsx', 2024),
    '2025-26': ('https://www.sportsbookreviewsonline.com/scoresoddsarchives/nhl/nhl%20odds%202025-26.xlsx', 2025),
}

all_rows = []

for label, (url, start_year) in SEASONS.items():
    print(f"\nFetching {label}...")
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        content = r.content

        # Try HTML first (SBR often serves HTML disguised as xlsx)
        raw = None
        try:
            tables = pd.read_html(content)
            if tables:
                raw = tables[0]
                print(f"  Parsed as HTML table: {raw.shape}")
        except Exception:
            pass

        # Try openpyxl
        if raw is None:
            try:
                from io import BytesIO
                raw = pd.read_excel(BytesIO(content), engine='openpyxl')
                print(f"  Parsed as xlsx (openpyxl): {raw.shape}")
            except Exception:
                pass

        # Try xlrd
        if raw is None:
            try:
                from io import BytesIO
                raw = pd.read_excel(BytesIO(content), engine='xlrd')
                print(f"  Parsed as xls (xlrd): {raw.shape}")
            except Exception:
                pass

        if raw is None:
            print(f"  ⚠️  Could not parse — content type: {r.headers.get('Content-Type')}")
            print(f"  First 200 bytes: {content[:200]}")
            continue

        print(f"  Columns: {list(raw.columns)}")
        print(f"  Sample:\n{raw.head(4).to_string()}")

        # ── Parse rows ────────────────────────────────────────────────────────
        cols = [str(c).strip() for c in raw.columns]
        raw.columns = cols

        # Find columns
        date_col  = next((c for c in cols if 'date' in c.lower()), None)
        team_col  = next((c for c in cols if 'team' in c.lower()), None)
        final_col = next((c for c in cols if 'final' in c.lower()), None)
        open_col  = next((c for c in cols if c.lower() == 'open'), None)
        close_col = next((c for c in cols if c.lower() in ('close','closer')), None)
        rot_col   = next((c for c in cols if 'rot' in c.lower()), None)

        print(f"  date={date_col} team={team_col} final={final_col} open={open_col} close={close_col}")

        if not team_col or not final_col:
            print(f"  ⚠️  Required columns not found. All cols: {cols}")
            continue

        if date_col:
            raw[date_col] = raw[date_col].ffill()

        rows = []
        i = 0
        while i < len(raw) - 1:
            r1, r2 = raw.iloc[i], raw.iloc[i+1]
            try:
                if rot_col and pd.notna(r1[rot_col]):
                    rot1 = int(float(str(r1[rot_col]).replace(',','')))
                    away_row = r1 if rot1 % 2 == 1 else r2
                    home_row = r2 if rot1 % 2 == 1 else r1
                else:
                    away_row, home_row = r1, r2

                # Parse date
                dval = str(away_row[date_col]).strip() if date_col else ''
                if not dval or dval in ('nan','NaT'):
                    i += 2; continue
                try:
                    if '/' in dval and len(dval) <= 6:
                        parts = dval.split('/')
                        mo, dy = int(parts[0]), int(parts[1])
                        yr = start_year if mo >= 8 else start_year + 1
                        date_str = f'{yr}-{mo:02d}-{dy:02d}'
                    else:
                        date_str = pd.to_datetime(dval).strftime('%Y-%m-%d')
                except:
                    i += 2; continue

                away = norm(away_row[team_col])
                home = norm(home_row[team_col])
                ascore = pd.to_numeric(away_row[final_col], errors='coerce')
                hscore = pd.to_numeric(home_row[final_col], errors='coerce')
                if pd.isna(ascore) or pd.isna(hscore):
                    i += 2; continue

                # Get O/U line — look for value between 4.5 and 9.5
                ou = np.nan
                for col in [close_col, open_col]:
                    if not col: continue
                    for row in [away_row, home_row]:
                        val = pd.to_numeric(row[col], errors='coerce')
                        if pd.notna(val) and 4.5 <= val <= 9.5:
                            ou = float(val); break
                    if pd.notna(ou): break

                open_line = np.nan
                if open_col:
                    for row in [away_row, home_row]:
                        val = pd.to_numeric(row[open_col], errors='coerce')
                        if pd.notna(val) and 4.5 <= val <= 9.5:
                            open_line = float(val); break

                rows.append({
                    'date': date_str, 'home_team': home, 'away_team': away,
                    'home_score': float(hscore), 'away_score': float(ascore),
                    'actual_total': float(hscore)+float(ascore),
                    'open_line': open_line, 'close_line': ou, 'season': label,
                })
            except Exception as ex:
                pass
            i += 2

        sdf = pd.DataFrame(rows)
        print(f"  Parsed: {len(sdf)} games | lines found: {sdf['close_line'].notna().sum()}")
        if len(sdf): all_rows.append(sdf)

    except requests.exceptions.HTTPError as e:
        print(f"  HTTP {e.response.status_code} — may not be posted yet")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")

if not all_rows:
    print("\n⚠️  Could not auto-download. Go to these URLs in your browser and save:")
    for label, (url, _) in SEASONS.items():
        print(f"  {label}: {url}")
    print("\nSave files to C:\\NHL\\data\\ as:")
    print("  nhl_lines_2023.xlsx  (or .xls or .html)")
    print("  nhl_lines_2024.xlsx")
    print("  nhl_lines_2025.xlsx")
    print("\nThen paste the first few rows here and we'll parse them.")
else:
    out = pd.concat(all_rows, ignore_index=True).sort_values('date').reset_index(drop=True)
    out.to_csv('data/historical_lines.csv', index=False)
    print(f"\n✅ Saved data/historical_lines.csv — {len(out)} games")
    print(f"Line coverage: {out['close_line'].notna().mean():.1%}")
    print(f"\nLine distribution:")
    vc = out['close_line'].value_counts().sort_index()
    for line, cnt in vc.items():
        over = (out[out['close_line']==line]['actual_total'] > line).mean()
        print(f"  {line}: {cnt} games | {over:.1%} went over")
    print("\nNext: python backtest_ou.py")