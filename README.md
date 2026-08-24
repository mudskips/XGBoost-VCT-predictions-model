# VCT 2025 Match Outcome Predictor

Predicting professional Valorant (VCT) match winners with an XGBoost model against a strong baseline (vlr.gg-style team ratings calculated based on team performance and stats).

## Results

Model achieved 65.3% accuracy predicting the winner of a given match vs. a 63.1% baseline on 504 regional-season matches (Stage 1 + Stage 2). Across all four eligible tournament phases in 2025 (Stage 1, Masters Toronto, Stage 2, Champions — 612 matches, walk-forward validated), the model had an overall accuracy of 63.4%, while vlr.gg-style ratings scored 62.4%.

The model saw decreased performance specifically at the two LAN/big-stage events, Masters Toronto and Champions, where it loses to the baseline (54.6% vs. 59.3%) even though it wins comfortably during the regional season (65.3% vs. 63.1%). See Limits below for why.

## What the model uses

Round-margin momentum (how much a team has been winning/losing by, averaged over its last 5 maps) drives most of the model's decisions, aided by round win rates based on economy (pistol, eco, and full-buy rounds). No player ratings, ADR, KAST, or any other player-level stat goes into the model at all.

That exclusion is actually what, to me, made the overall result interesting. This model (slightly) beats out a rating-based baseline using economy data alone. I tested feeding the model the rating/ADR features directly, both on their own and combined with the economy features, but on their own they land at about the same accuracy as the economy-only model, and combined, accuracy actually dropped (61.8% vs. 63.4%). My results suggest that recent round-level economic execution carries roughly as much predictive signal as player-based stats and rating for these matches, which I found surprising!

## Data

All raw data comes from [Ryan Luong's VCT Kaggle
dataset](https://www.kaggle.com/datasets/ryanluong1/valorant-champion-tour-2021-2023-data)
(2025 folder): per-player per-map stats (`overview25.csv`), per-map scores
(`maps_scores25.csv`), and round-economy outcomes (`eco_stats.csv`). Economy data has no China-region coverage, so China matches are
excluded from the model (they're still counted in the baseline where rating
data is available).

## Methodology

- Every rolling feature uses `shift(1)` before the window so a match's own result does not influence its own prediction. 
- The model is retrained at
each tournament phase boundary using only matches from before that phase, then evaluated on the current matches.
- Both are scored on identical games across all four eligible phases.
- XGBoost (`max_depth=2`, heavy L1/L2 regularization, `min_child_weight=12`), which is deliberately shallow and regularized given the
dataset is pretty small for something like this.

## Limits

Data for official VCT matches is obviously a relatively small evaluation set for training, so that should be kept in mind in regards to the model's edge. The more interesting limit: the model consistently loses to the baseline at Masters Toronto and Champions specifically, even though it wins during the regular season. My best guess is that round-level economic form over the last 5 maps is a noisier signal at LAN majors — teams face unfamiliar opponents from other regions for the first time, travel and patch familiarity vary, and the fold sizes are small (42 and 66 matches), so a handful of upsets swings the number a lot. vlr.gg's rating, built from broader player performance data over a longer history, may just hold up better in exactly that unfamiliar-matchup setting than a purely round-economy signal does.

## Running

```
pip install pandas numpy xgboost scikit-learn
python3 cleaning.py     # raw Kaggle CSVs -> data_clean/*.csv
jupyter notebook main.ipynb   # walk-forward training + evaluation
```
