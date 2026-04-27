# NHL Prediction Model — Project Context
*Last updated: April 19, 2026*

---

## What We Built

A full NHL prediction model running locally in VS Code at `C:\NHL`. It pulls live data daily, makes moneyline, O/U, and player prop picks, tracks record automatically, and displays everything in a clean HTML dashboard. Now includes playoff support, season phase features, goalie GSAx in training, and historical odds backtesting.

---

## Tech Stack

- **Python 3.12** — no virtual environment
- **Key libraries:** pandas, scikit-learn, xgboost, joblib, requests, scipy
- **Data sources:** NHL API, MoneyPuck CSVs, DailyFaceoff (starters), ESPN (injuries), The Odds API (odds + props), checkbestodds.com (historical ML odds)
- **Output:** `picks.html` — opens in browser, has ML, O/U, and Props tabs

---

## File Structure

```
C:\NHL\
├── data/
│   ├── games.csv                  # current season games (2025-26)
│   ├── games_2324.csv             # 2023-24 season games
│   ├── games_2425.csv             # 2024-25 season games
│   ├── moneypuck.csv              # current season team stats (5on5, PP, PK)
│   ├── mp_2023.csv                # 2023-24 MoneyPuck team stats
│   ├── mp_2024.csv                # 2024-25 MoneyPuck team stats
│   ├── goalies.csv                # NHL API goalie stats (current season)
│   ├── goalies_mp.csv             # MoneyPuck goalie GSAx data
│   ├── skaters_mp.csv             # MoneyPuck skater data (injury impact + props)
│   ├── standings.csv              # current standings
│   ├── training_data.csv          # combined 3-season training dataset (59 features)
│   ├── model.pkl                  # trained moneyline model
│   ├── model_total.pkl            # trained O/U regression model
│   ├── record.json                # all-time pick record (ML + O/U)
│   ├── team_goalie_gsax.csv       # team-season weighted GSAx lookup
│   ├── historical_lines.csv       # DraftKings O/U lines saved daily (auto-populated)
│   ├── historical_ml_odds.csv     # parsed ML odds from checkbestodds.com (2025-26)
│   └── raw_odds.txt               # raw odds text from checkbestodds.com
├── fetch_data.py                  # fetch standings
├── fetch_games.py                 # fetch current season games
├── fetch_all_seasons.py           # fetch 2023-24 and 2024-25 games
├── fetch_historical_mp.py         # fetch historical MoneyPuck CSVs
├── fetch_historical_goalies.py    # fetch team-season GSAx from MoneyPuck + NHL API
├── fetch_playoff_games.py         # fetch historical playoff games → training_data
├── build_historical_features.py   # build training_data.csv (3 seasons)
├── add_goalie_training.py         # join team GSAx to training_data by date+team
├── add_season_phase.py            # add season_phase, season_progress, is_playoffs
├── parse_historical_ml.py         # parse checkbestodds.com tab-delimited odds → CSV
├── model.py                       # train moneyline model → model.pkl
├── model_total.py                 # train O/U regression model → model_total.pkl
├── get_starters.py                # scrape DailyFaceoff for today's starters
├── get_injuries.py                # pull ESPN injury API
├── get_odds.py                    # pull The Odds API (ML + totals)
├── get_props.py                   # pull The Odds API player props (shots, saves, points)
├── grade.py                       # auto-grade yesterday's picks from boxscores
├── build_html.py                  # MAIN SCRIPT — generates picks.html
├── backtest.py                    # ML model backtest (train 2023-25, test 2025-26)
├── backtest_ou.py                 # O/U model backtest vs fixed and real lines
├── picks.html                     # today's picks dashboard
└── picks_YYYY-MM-DD.html          # historical pick pages (improved UI)
```

---

## Daily Workflow

Run every day before games start:

```bash
python build_html.py
start picks.html
```

`build_html.py` automatically:
1. Auto-grades yesterday's picks (ML + O/U) from NHL boxscores
2. Syncs actual scores to historical_lines.csv
3. Fetches standings, starters, injuries, odds, and player props
4. Makes ML, O/U, and prop predictions for tonight
5. Sets season_phase=5, is_playoffs=1 automatically during playoffs
6. Saves picks and tonight's DraftKings lines to record.json and historical_lines.csv
7. Generates picks.html and history pages

---

## Full Reset / Retrain Sequence (start of each season)

```bash
python fetch_games.py
python fetch_historical_mp.py
python build_historical_features.py
python fetch_historical_goalies.py
python add_goalie_training.py
python add_season_phase.py
python fetch_playoff_games.py
python model.py
python model_total.py
python build_html.py
```

---

## Model Performance

### Moneyline Model
- **Algorithm:** Logistic Regression (beats RF, XGBoost, GBM on this data)
- **Training data:** 3,188 games across 2023-24, 2024-25, 2025-26 + playoff games
- **Cross-val accuracy:** 61.8%
- **Backtest (2025-26 season out-of-sample, last 20%):**
  - 60-65% conf: 64.1% accuracy, +22.4% ROI
  - 65-70% conf: 71.7% accuracy, +36.9% ROI
  - 70%+ conf: 79.4% accuracy, +51.6% ROI
- **Live record:** 25-17 (59.5%) as of April 19, 2026

### Key Features (moneyline)
```
sv_xg_pct_diff     # score-adjusted xG% differential (top feature)
pdo_diff           # luck/regression stat differential
home_sv_xg_pct     # home team score-adjusted xG%
xg_pct_diff        # expected goals % differential
fenwick_pct_diff   # unblocked shot attempt %
home_xg_pct        # home expected goals %
home_pdo           # home PDO
gsax_diff          # goalie GSAx differential (team-season weighted)
away_pdo           # away PDO
hdc_pct_diff       # high danger chance % differential
home_hdc_pct       # home high danger chance %
corsi_pct_diff     # shot attempt % differential
```

### Post-model adjustments (in build_html.py)
- Home/road win rate adjustment ±8%
- Back-to-back penalty ±4%
- Last 10 games form adjustment ±6%
- Game importance: eliminated teams get -6% when opponent has playoff stakes
- Goalie GSAx adjustment ±0.05% per GSAx point (reduced — model now handles main weight)
- Injury impact (based on injured player's gameScore per game)
- **Season phase:** set automatically — is_playoffs=1, season_phase=5 during playoffs

### Betting Strategy
- **Below 60%** → Skip entirely (no edge in backtest)
- **60-65%** → Small bet
- **65-70%** → Standard bet
- **70%+** → Max bet

---

## O/U Model

### How it works
- **Algorithm:** GradientBoostingRegressor (MAE ~1.85 goals)
- **Predicts:** Raw total goals (e.g. 6.2)
- **At prediction time:** Compares to DraftKings line using normal CDF
- `P(over) = 1 - norm.cdf(book_line, loc=pred_total, scale=residual_std)`
- **residual_std:** ~2.1 goals

### Key features (O/U)
```
home_sv_xgf, home_sv_xga   # score-adjusted xG for/against
away_sv_xgf, away_sv_xga
home_hdcf, home_hdca        # high danger chances
away_hdcf, away_hdca
home_pdo, away_pdo
```

### O/U Backtest (vs fixed 6.5 line)
- 60-65% conf: 59.1% accuracy, +12.7% ROI
- 65-70% conf: 53.9% accuracy, +2.9% ROI
- 70%+ conf: 54.3% accuracy, +3.8% ROI
- **Only bet 60%+ confidence**

### O/U known issue
The model predicts mostly unders against the 6.5 line (131 over vs 507 under in backtest). This is partly structural — late season 6.5 lines are set slightly high. Monitor live record before trusting heavily.

### historical_lines.csv (auto-populated going forward)
- Saves tonight's DraftKings O/U lines every day automatically
- Fills in actual scores the next morning after grading
- Will enable proper real-line O/U backtesting after enough data accumulates

---

## Season Phase Features

Added to training_data and set automatically in build_html.py:

| Phase | Label | When |
|---|---|---|
| 1 | Early | First ~15% of game days (Oct-Nov) |
| 2 | Mid | 15-45% of game days (Nov-Jan) |
| 3 | Late | 45-70% of game days (Jan-Mar) |
| 4 | Crunch | 70-100% of game days (Mar-Apr) |
| 5 | Playoffs | April 19+ |

- `season_progress`: 0.0 → 1.0 through regular season, 1.1 for playoffs
- `is_playoffs`: binary flag

---

## Goalie GSAx in Training

Team-season weighted average GSAx joined to training_data by date + team.

**Source:**
- 2025-26: MoneyPuck season summary (public)
- 2023-24, 2024-25: NHL API (`savePct - 0.906) × shotsAgainst` approximation

**File:** `data/team_goalie_gsax.csv`
- 108 rows (3 seasons × ~32 teams + some traded players with combined entries like `ANA,BUF`)
- 100% match rate against training_data

---

## Playoff Data in Training

Historical playoff games (gameType=3) from 2023-24 and 2024-25 fetched via NHL API and appended to training_data with `is_playoffs=1`, `season_phase=5`.

- 2023-24: 88 playoff games
- 2024-25: ~88 playoff games
- Total playoff games in training: ~184

---

## Historical ML Odds

**Source:** checkbestodds.com — 3-way ML (home/draw/away) decimal odds
**File:** `data/historical_ml_odds.csv`
**Coverage:** Oct 2025 – Apr 2026 (1,280 games)
**Format after parsing:** date, home_team (abbr), away_team (abbr), home_odds_dec, draw_odds_dec, away_odds_dec, home_implied, away_implied

**To update for next season:**
1. Go to checkbestodds.com/hockey-odds/archive-nhl
2. Copy the full season odds text
3. Save to `data/raw_odds.txt`
4. Run `python parse_historical_ml.py`

**Note:** The odds use European format with a draw option (hockey has OT). The implied probabilities are calculated with vig removal across all three outcomes.

---

## HTML Dashboard

`picks.html` has **three tabs**:
1. **Moneyline** — sorted by EV, shows odds/book implied/model%/EV per card
2. **Over/Under** — sorted by confidence, shows O/U pick + predicted total vs book line
3. **Props** — sorted by EV, grouped by market (Shots → Saves → Points)

**History pages** (picks_YYYY-MM-DD.html) now feature:
- ML and O/U W-L summary boxes for the day
- Sortable picks (by confidence, ML result, O/U result, or EV)
- ML / O/U tab toggle
- W/L badges with final scores and totals
- Color-coded EV

**Playoff label:** Shows "NHL Playoffs 2026" instead of playoff race labels during playoffs.

---

## Player Props

- **`get_props.py`** fetches props from The Odds API per game event
- Each market fetched in a separate API call
- **Model edge:** Poisson distribution using MoneyPuck per-game rates vs book implied probability
- Shown in Props tab, sorted by EV

### Valid NHL prop markets
```
player_shots_on_goal    ✓
player_goalie_saves     ✓
player_points           ✓
player_points_assists   ✓
```

---

## API Keys

### The Odds API
```
Key: a62a46c2a2679e8ce805ecf918c948c0
ML + totals: https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds/
Props:       https://api.the-odds-api.com/v4/sports/icehockey_nhl/events/{id}/odds
```

### Anthropic API
```
Key: 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
```

---

## Record Tracking

`record.json` stores:
- `alltime` — overall ML W/L
- `by_conf` — ML record by confidence band (50/55/60/65/70%)
- `by_month` — ML monthly breakdown
- `by_day` — daily picks with individual game results
- `ou_alltime` — overall O/U W/L
- `ou_by_conf` — O/U by confidence band
- `ou_by_month` — O/U monthly breakdown

Auto-grading checks NHL boxscores each morning and updates all records.

---

## Backtesting

### ML Backtest (`backtest.py`)
- Train: training_data seasons 2023 + 2024
- Test: last 20% of training_data (2025-26 season not yet in training_data)
- Uses real decimal odds from historical_ml_odds.csv when available
- Includes CLV check (does model beat closing line implied probability?)

### O/U Backtest (`backtest_ou.py`)
- Tests against fixed lines (5.5, 6.0, 6.5, 7.0) and real lines when available
- historical_lines.csv will accumulate real lines automatically going forward

---

## Known Issues / TODO

### In Progress
- **Real odds for backtest:** historical_lines.csv auto-saves going forward; real O/U backtesting will improve as data accumulates
- **Series context for playoffs:** teams playing differently based on series score (0-1 vs 1-0) not yet modeled

### Improvements for Next Season (October 2026)
1. **Rolling form windows** — last 5/10 game xG%, Corsi, PDO as features (biggest remaining improvement)
2. **Rest days as training features** — replace flat B2B adjustment with learned feature
3. **Home/road splits** — home-only and road-only xG% instead of aggregate
4. **Prop record tracking** — save prop picks to record.json and grade nightly
5. **CLV logging** — save opening odds at pick time to enable closing line value analysis
6. **Head-to-head features** — last 3 H2H results as a feature
7. **Add 2025-26 playoff games** — run fetch_playoff_games.py after Stanley Cup Finals end

### Bugs Fixed This Project
- O/U model predicting 83.7% overs → fixed by switching to regression approach
- MoneyPuck per-game goalie files (403) → use season summary + NHL API instead
- training_data had no game_id → join goalie GSAx by date instead
- training_data stores full team names, goalie GSAx used abbrs → added FULL_TO_ABBR map
- `fetch_playoff_games.py` crashed on median of string column → skip non-numeric cols
- `from build_historical_features import load_mp` ran entire script → define load_mp locally
- Season phase assigned wrong (month >= 5 caught regular season) → use April 19+ cutoff
- build_html found 0 games during playoffs → added gameType==3 filter
- History pages showed minimal info → rebuilt with sortable cards, W/L badges, tabs

---

## Where We Left Off

**April 19, 2026 — Playoffs just started**

Model status:
- ✅ ML — working, 25-17 (59.5%) live record, strong backtest at 65%+
- ✅ O/U — working, 6-6 live (small sample), +12.7% ROI at 60%+ in backtest
- ✅ Props — working, showing in Props tab
- ✅ Playoff support — season_phase=5, is_playoffs=1 set automatically
- ✅ History pages — rebuilt with sortable UI
- ✅ Historical ML odds — parsed 1,280 games from checkbestodds.com
- ✅ historical_lines.csv — auto-saves DraftKings lines daily going forward

**Next tasks:**
1. Series context tracker for playoffs (team record in current series)
2. Rolling form window features for next season retraining
3. Update raw_odds.txt with playoff odds from checkbestodds.com periodically
