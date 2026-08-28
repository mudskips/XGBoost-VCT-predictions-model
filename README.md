# VCT 2025 Match Outcome Predictions with ML vs Stats

In this project I developed an XGBoost model and team ratings calculated based on team performance and stats from previous matches, all with the goal of seeing which can better predict which team will win a game of professional VALORANT.

## Motivation

I've been interested in pro Valorant since I started playing in 2021. I wanted to find a way to combine my interests in math, ML, and valorant, and this project seemed like a perfect combination of the 3. My next step would be to expand the data to include 2026 and present matches, but I'mm gonna wait until after champions for that.

## Results

The XGBoost model achieved 64.9% accuracy predicting the winner of a given match vs. 63.1% for the hand-calculated ratings on 504 regional-season matches (Stage 1 + Stage 2). Across all four eligible tournament phases in 2025 (Stage 1, Masters Toronto, Stage 2, Champions), the model had an overall accuracy of 63.4%, while choosing a winner based on the ratings calculations yielded aam ccuracy of 62.4%.

## Limitations

The model saw decreased performance specifically at the two big LAN events, Masters Toronto and Champions, where it loses to ratings (54.6% vs. 59.3%). My best guess for why is that round-level economic form over the last 5 maps is a weaker sign of future performance at LAN majors, as the world's best teams face unfamiliar opponents from other regions and the number of total matches is small (42 and 66 matches), so upsets (which are way more common in esports) change the accuracy a lot. Calculated team ratings built from broader player performance data over a longer history also may just hold up better under the pressure and high-caliber competition of LAN tournaments than an ML model trained on less than 1000 games can. Additionally, data for official VCT matches is obviously a relatively small set for this kind of training, so that should be kept in mind in regards to the model's ~2% edge over ratings. 

## How I trained the model and calculated the ratings

The model primarily used round-margin momentum (how much a team has been winning/losing by, averaged over its last 5 maps), aided by round win rates based on economy (pistol, eco, and full-buy rounds). No stats like  ADR, KAST, or any other low level stats contributed to the model's predictions.

For the team-ratings, I used per-player statistics and ratings as well as shrinkage-blending and rolled them up into my own rolling team-ratings. To see the exact math, see main and cleaning.py.

## On the data

The data I used comes from Ryan Luong's VCT Kaggle
dataset(https://www.kaggle.com/datasets/ryanluong1/valorant-champion-tour-2021-2023-data)
(2025 folder): per-player per-map stats (`overview25.csv`), per-map scores
(`maps_scores25.csv`), and round-economy outcomes (`eco_stats.csv`). Economy data is missing for the China region, so China matches are
excluded from the model (they're still counted in the team-ratings where data IS available).

## Methods

- Rolling features use `shift(1)` before the window to avoid leakage. 
- Model is retrained at each tournament phase boundary using  matches from before that phase.
- Model and ratings are scored on identical games.
- I used XGBoost with (`max_depth=2`, heavy L1/L2 regularization, `min_child_weight=12`) given the
dataset is relatively small

## How to run

```
pip install pandas numpy xgboost scikit-learn
python3 cleaning.py     # raw Kaggle CSVs -> data_clean/*.csv
jupyter notebook main.ipynb   # walk-forward training + evaluation
```
