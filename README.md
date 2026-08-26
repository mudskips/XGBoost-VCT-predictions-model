# VCT 2025 Match Outcome Predictor

Predicting professional Valorant (VCT) match winners with an XGBoost model against a strong baseline (vlr.gg-style team ratings calculated based on team performance and stats)

## Results

Model achieved 64.9% accuracy predicting the winner of a given match vs. a 63.1% baseline on 504 regional-season matches (Stage 1 + Stage 2). Across all four eligible tournament phases in 2025 (Stage 1, Masters Toronto, Stage 2, Champions), the model had an overall accuracy of 63.4%, while vlr.gg-style ratings scored 62.4%.

The model saw decreased performance specifically at the two big LAN events, Masters Toronto and Champions, where it loses to the baseline (54.6% vs. 59.3%) even though it wins comfortably during the regional season (65.3% vs. 63.1%). See "Where it falls short" below for why I think this might've happened.

## How I trained the model

The model primarily used round-margin momentum (how much a team has been winning/losing by, averaged over its last 5 maps) drives most of the model's decisions, aided by round win rates based on economy (pistol, eco, and full-buy rounds). No stats like  ADR, KAST, or any other player-level stat went into this model.

That exclusion is what, to me, made the overall result super interesting. This model (slightly) beats out a rating-based baseline using just economy data. I tried feeding the model the rating/ADR features too, both on their own and combined with the economy features, but on their own they land at about the same accuracy as the economy-only model, and combined, accuracy actually dropped (61.8% vs. 63.4%). My results suggest that recent round-level economic execution carries roughly as much predictive signal as player-based stats and rating for these matches, which I found surprising.

### Something to note about the baseline

For this project, I had initially use per-player statistics and ratings as well as shrinkage-blending and rolled them up into my own rolling team-ratings to feed into the model, however I found this actually made the model perform slightly worse. Why? Don't ask me. I did, however, notice that these personally calculated team ratings were slightly better ( ~0.7%) at predicting outcomes, so I decided to use them as a baseline.

## On the data

Th data I used comes from Ryan Luong's VCT Kaggle
dataset(https://www.kaggle.com/datasets/ryanluong1/valorant-champion-tour-2021-2023-data)
(2025 folder): per-player per-map stats (`overview25.csv`), per-map scores
(`maps_scores25.csv`), and round-economy outcomes (`eco_stats.csv`). Economy data is missing for the China region, so China matches are
excluded from the model (they're still counted in the baseline where rating
data IS available).

## Methods

- Every rolling feature uses `shift(1)` before the window to avoid leakage. Additionally, The model is retrained at each tournament phase boundary using  matches from before that phase.
- Model and baseline are scored on identical games. Baseline represents picking whichever team in a head-to-head matchup has the higher rating, while the model utilizes extreme gradient boosting (mentioned next) 
- Uses XGBoost with (`max_depth=2`, heavy L1/L2 regularization, `min_child_weight=12`) given the
dataset is relatively small.

## Where it falls short

Data for official VCT matches is obviously a relatively small set for this kind of training, so that should be kept in mind in regards to the model's ~2% edge. Another thing to note is that the model consistently loses to the baseline at Masters Toronto and Champions, even though it wins during the regular season. My best guess is that round-level economic form over the last 5 maps is a weaker sing of future performance at LAN majors, as the world's best teams face unfamiliar opponents from other regions for the first time, and the fold sizes are small (42 and 66 matches), so upsets (which are way more common in esports) can change the accuracy a lot. Calculated team ratings built from broader player performance data over a longer history also may just hold up better under pressure.

## How to run

```
pip install pandas numpy xgboost scikit-learn
python3 cleaning.py     # raw Kaggle CSVs -> data_clean/*.csv
jupyter notebook main.ipynb   # walk-forward training + evaluation
```
