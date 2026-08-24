import pandas as pd
import numpy as np

df_overview = pd.read_csv("data_raw/overview25.csv")
df_map = pd.read_csv("data_raw/maps_scores25.csv")

#filter out per-side and all-map columns, remove special characters, fix blank cells
df_overview = df_overview[df_overview['Side'] == 'both']
df_overview = df_overview[df_overview['Map'] != 'All Maps']
df_overview['Kill, Assist, Trade, Survive %'] = df_overview['Kill, Assist, Trade, Survive %'].str.replace('%', '', regex = False)
df_overview['Headshot %'] = df_overview['Headshot %'].str.replace('%', '', regex = False)
for col in ['Rating', 'Average Combat Score', 'Average Damage Per Round', 'Kill, Assist, Trade, Survive %', 'Headshot %']:
    df_overview[col] = pd.to_numeric(df_overview[col], errors='coerce')

Tournament_order = {
    "VCT 2025: China Kickoff": 1, 
    "VCT 2025: Americas Kickoff": 1,
    "VCT 2025: Pacific Kickoff": 1, 
    "VCT 2025: EMEA Kickoff": 1,
    "Valorant Masters Bangkok 2025": 2,
    "VCT 2025: Americas Stage 1": 3, 
    "VCT 2025: China Stage 1": 3,
    "VCT 2025: Pacific Stage 1": 3, 
    "VCT 2025: EMEA Stage 1": 3,
    "Valorant Masters Toronto 2025": 4,
    "VCT 2025: China Stage 2": 5, 
    "VCT 2025: Pacific Stage 2": 5,
    "VCT 2025: EMEA Stage 2": 5, 
    "VCT 2025: Americas Stage 2": 5,
    "Valorant Champions 2025": 6,
}

df_map['row_in_file'] = np.arange(len(df_map))
df_map['tourney_phase'] = df_map['Tournament'].map(Tournament_order)
df_map = df_map.sort_values(['tourney_phase', 'row_in_file']).reset_index(drop=True)
df_map['order'] = np.arange(len(df_map))

df_map['A_Win'] = df_map['Team A Score'] > df_map['Team B Score']

map_key = ['Tournament', 'Stage', 'Match Type', 'Match Name', 'Map']
keep_cols = map_key + ['tourney_phase', 'order', 'Team A', 'Team B', 'Team A Score', 'Team B Score', 'A_Win']

copy_a = df_map[keep_cols].rename(columns={
    'Team A': 'Team', 'Team B': 'Opponent',
    'Team A Score': 'team_score', 'Team B Score': 'opp_score',
})

copy_a['won'] = copy_a['A_Win'].astype(int)

copy_b = df_map[keep_cols].rename(columns={
    'Team B': 'Team', 'Team A': 'Opponent',
    'Team B Score': 'team_score', 'Team A Score': 'opp_score',
})

copy_b['won'] = (~copy_b['A_Win']).astype(int)

results = pd.concat([copy_a, copy_b], ignore_index=True).drop(columns='A_Win')

df_overview = df_overview.merge(results, on=map_key + ['Team'], how='left')

df_overview = df_overview.sort_values('order').reset_index(drop=True)

stats = {
    'rating': 'Rating', 
    'acs': 'Average Combat Score', 
    'adr': 'Average Damage Per Round',
    'hs': 'Headshot %', 
    'kast': 'Kill, Assist, Trade, Survive %',
    'fkd': 'Kills - Deaths (FKD)', 
    'won': 'won',
}

#roll up player history (shift(1) always excludes the current map to avoid leakage)
g = df_overview.groupby('Player', sort=False)
for s, col in stats.items():
    df_overview[f'p_{s}_career'] = g[col].transform(lambda x: x.shift(1).expanding().mean())
    df_overview[f'p_{s}_last5']  = g[col].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
df_overview['p_maps_played_prior'] = g[stats['rating']].transform(lambda x: x.shift(1).expanding().count())

# league-wide fallback for new players, bucketed by 'order' so
#   simultaneous matches can't leak into each other
bucket = df_overview.groupby('order')[[stats[s] for s in stats]].agg(['sum', 'count'])
bucket.columns = ['_'.join(c) for c in bucket.columns]
bucket = bucket.sort_index()
prior_sum = bucket.filter(like='_sum').cumsum().shift(1)
prior_cnt = bucket.filter(like='_count').cumsum().shift(1)
league_avg = pd.DataFrame(prior_sum.values / prior_cnt.values,
                           columns=[c.replace('_sum', '') for c in prior_sum.columns], index=bucket.index)
league_avg = league_avg.rename(columns={stats[s]: f'league_{s}' for s in stats}).reset_index()
df_overview = df_overview.merge(league_avg, on='order', how='left')

# shrinkage blend: own rolling stats vs. teammate/league fallback
# K is a tunable number to determine minimum games needed to avoid league avg fallback
K = 8
team_map_key = map_key + ['Team']
for s in stats:
    for w in ['career', 'last5']:
        col = f'p_{s}_{w}'
        grp = df_overview.groupby(team_map_key)[col]
        team_sum, team_cnt = grp.transform('sum'), grp.transform('count')
        self_val, self_present = df_overview[col].fillna(0), df_overview[col].notna().astype(int)
        teammate_avg = (team_sum - self_val) / (team_cnt - self_present).replace(0, np.nan)
        fallback = teammate_avg.fillna(df_overview[f'league_{s}'])
        own = df_overview[col]
        weight = (df_overview['p_maps_played_prior'] / K).clip(upper=1.0)
        df_overview[f'eff_{s}_{w}'] = np.where(own.notna(), weight * own.fillna(0) + (1 - weight) * fallback, fallback)

# roll the 5-player roster up to team-match rows
team_roll = df_overview.groupby(team_map_key, as_index=False, sort=False).agg(
    **{f'team_{s}_{w}_avg': (f'eff_{s}_{w}', 'mean') for s in stats for w in ['career', 'last5']},
    team_roster_experience=('p_maps_played_prior', 'mean'),
    order=('order', 'first'), won=('won', 'first'),
    team_score=('team_score', 'first'), opp_score=('opp_score', 'first'), Opponent=('Opponent', 'first'),
)
team_roll = team_roll.sort_values('order').reset_index(drop=True)
team_roll['map_id'] = team_roll.groupby(map_key, sort=False).ngroup()

# team-level on-map and head-to-head history (unchanged approach from before)
g_onmap = team_roll.groupby(['Team', 'Map'], sort=False)
team_roll['team_onmap_won_career_avg'] = g_onmap['won'].transform(lambda s: s.shift(1).expanding().mean())
team_roll['team_onmap_won_last5_avg']  = g_onmap['won'].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
team_roll['team_onmap_maps_played_prior'] = g_onmap['won'].transform(lambda s: s.shift(1).expanding().count())

g_h2h = team_roll.groupby(['Team', 'Opponent'], sort=False)
team_roll['h2h_won_career_avg'] = g_h2h['won'].transform(lambda s: s.shift(1).expanding().mean())
team_roll['h2h_maps_played_prior'] = g_h2h['won'].transform(lambda s: s.shift(1).expanding().count())

# join + diffs
feature_cols = [c for c in team_roll.columns if c.startswith('team_') and c.endswith(('_avg', 'experience'))]
opp = team_roll[['map_id', 'Team'] + feature_cols].copy()
opp.columns = ['map_id', 'Opponent'] + [f'opp_{c}' for c in feature_cols]
final = team_roll.merge(opp, on=['map_id', 'Opponent'], how='left', validate='one_to_one')
for c in feature_cols:
    final[f'diff_{c}'] = final[c] - final[f'opp_{c}']

meta_cols = ['map_id', 'order', 'Tournament', 'Stage', 'Match Type', 'Match Name', 'Map', 'Team', 'Opponent', 'team_score', 'opp_score']
model_feature_cols = [c for c in final.columns if c.startswith('diff_')] + ['h2h_won_career_avg', 'h2h_maps_played_prior', 'team_onmap_maps_played_prior']
training_table = final[meta_cols + model_feature_cols + ['won']].copy()
training_table.to_csv('data_clean/2025_training.csv', index=False)

#Eco cleaning

# load raw data
data = pd.read_csv('data_clean/2025_training.csv').sort_values('order').reset_index(drop=True)
eco = pd.read_csv('data_raw/eco_stats.csv')  # raw: one row per round per team
# raw eco columns: Tournament, Stage, Match Type, Match Name, Map, Team,
#   Type ('Pistol'/'Eco'/'$$$'), Won (0/1), Initiated (0/1)
# NOTE: eco_stats.csv has zero coverage for China (Kickoff/Stage 1/Stage 2) -- dropped below via inner merge.

match_key = ['Tournament', 'Stage', 'Match Type', 'Match Name']
mapkey = match_key + ['Map']

# STEP 1: round-level rows -> map-level win rates
# raw eco is one row per round; pivot it up to one row per (map, team)
piv_won = eco.pivot_table(index=mapkey + ['Team'], columns='Type', values='Won', aggfunc='sum').reset_index()
piv_init = eco.pivot_table(index=mapkey + ['Team'], columns='Type', values='Initiated', aggfunc='sum').reset_index()
piv_init = piv_init.rename(columns={c: c + '_init' for c in piv_init.columns if c not in mapkey + ['Team']})
eco_wide = piv_won.merge(piv_init, on=mapkey + ['Team'])

eco_wide['pistol_win_rate'] = eco_wide['Pistol Won'] / 2.0  # always exactly 2 pistols per map
eco_wide['eco_win_rate'] = eco_wide['Eco (won)'] / eco_wide['Eco (won)_init'].replace(0, np.nan)
eco_wide['fullbuy_win_rate'] = eco_wide['$$$ (won)'] / eco_wide['$$$ (won)_init'].replace(0, np.nan)
eco_wide = eco_wide.fillna(0.5)  # team had 0 rounds of that type on this map -> neutral prior, not dropped

# STEP 2: attach chronological order, then roll into leak-safe rolling averages
order_lookup = data[mapkey + ['Team', 'order']].drop_duplicates()
eco_wide = eco_wide.merge(order_lookup, on=mapkey + ['Team'], how='inner')  # inner join drops China here
eco_wide = eco_wide.sort_values('order').reset_index(drop=True)

g = eco_wide.groupby('Team', sort=False)
for col in ['pistol_win_rate', 'eco_win_rate', 'fullbuy_win_rate']:
    # shift(1) BEFORE rolling -- a team's feature for map N only sees maps 1..N-1, never map N itself
    eco_wide[col + '_last5'] = g[col].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())

# STEP 3: select the columns that matter and export
out_cols = mapkey + ['Team', 'order', 'pistol_win_rate_last5', 'fullbuy_win_rate_last5', 'eco_win_rate_last5']
economy_features = eco_wide[out_cols].reset_index(drop=True)
economy_features.to_csv('economy_features.csv', index=False)