build_script = "build_" + "html" + ".py"

content = """\
name: Daily NHL Picks

on:
  schedule:
    - cron: '0 15 * * *'
  workflow_dispatch:

permissions:
  contents: write

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repo
        uses: actions/checkout@v4
        with:
          token: ${{ secrets.GITHUB_TOKEN }}

      - name: Set up Python 3.12
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install pandas scikit-learn joblib requests scipy

      - name: Run build_html.py
        env:
          ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: python build_html.py

      - name: Commit updated picks
        run: |
          git config --global user.name "github-actions[bot]"
          git config --global user.email "github-actions[bot]@users.noreply.github.com"
          git add picks.html data/record.json data/historical_lines.csv history/
          git diff --staged --quiet || git commit -m "Auto picks $(date +%Y-%m-%d)"
          git push
"""

content = content.replace("build_html.py", build_script)

with open(r'.github/workflows/daily_picks.yml', 'w', newline='\n') as f:
    f.write(content)

print("Done!")