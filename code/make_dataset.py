
import numpy as np
import pandas as pd

df = pd.read_csv('../data/ODI_ball_by_ball_updated.csv')

#print(df.loc[df['wicket_type'].notna()])

df['run_scored'] = (df['runs_off_bat'].fillna(0)+df['extras'].fillna(0)+df['wides'].fillna(0)+df['noballs'].fillna(0)+df['byes'].fillna(0)+df['legbyes'].fillna(0)).astype(int)

df['total_runs'] = df.groupby(['match_id', 'innings'])['run_scored'].transform('sum')

df['current_run'] = df.groupby(['match_id', 'innings'])['run_scored'].transform('cumsum')

df['runs_remaining'] = df['total_runs'] - df['current_run']

df['wicket_indicator'] = ((df['wicket_type'].notna()) & (df['wicket_type'] != 'retired hurt')).astype(int)

df['wickets_down'] = df.groupby(['match_id', 'innings'])['wicket_indicator'].transform('cumsum')

columns_to_drop = ['venue','runs_off_bat','extras','wides','noballs','byes','legbyes','penalty','player_dismissed','other_player_dismissed','other_wicket_type','striker' ,'non_striker','bowler','wicket_type','wicket_indicator']

df = df.drop(columns=columns_to_drop)
df.to_csv('../data/modified_data.csv', index=False)

print("done")