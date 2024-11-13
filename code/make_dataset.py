
import numpy as np
import pandas as pd


""" ODI dataset creation """

# df = pd.read_csv('../data/ODI_ball_by_ball_updated.csv')

# #print(df.loc[df['wicket_type'].notna()])

# df['runs_from_ball'] = (df['runs_off_bat'].fillna(0)+df['extras'].fillna(0)+df['wides'].fillna(0)+df['noballs'].fillna(0)+df['byes'].fillna(0)+df['legbyes'].fillna(0)).astype(int)

# df['total_runs'] = df.groupby(['match_id', 'innings'])['runs_from_ball'].transform('sum')

# df['team_score'] = df.groupby(['match_id', 'innings'])['runs_from_ball'].transform('cumsum')

# df['runs_remaining'] = df['total_runs'] - df['team_score']

# df['wicket_indicator'] = ((df['wicket_type'].notna()) & (df['wicket_type'] != 'retired hurt')).astype(int)

# df['wickets_down'] = df.groupby(['match_id', 'innings'])['wicket_indicator'].transform('cumsum')

# df[['over', 'ball']] = df['ball'].astype(str).str.split('.', expand=True)

# df['ball_no'] = df.groupby(['match_id', 'innings']).cumcount() + 1

# df['total_balls'] = df.groupby(['match_id', 'innings'])['wickets_down'].transform('size')

# df['balls_remaining'] = df['total_balls'] - df['ball_no']

# columns_to_drop = ['venue','runs_off_bat','extras','wides','noballs','byes','legbyes','penalty','player_dismissed','other_player_dismissed','other_wicket_type','striker' ,'non_striker','bowler','wicket_type','wicket_indicator']
# df = df.drop(columns=columns_to_drop)


# inning1_scores = df[df['innings'] == 1].set_index('match_id')['total_runs'].to_dict()
# df['target_score'] = -1 
# df.loc[df['innings'] == 2, 'target_score'] = df['match_id'].map(inning1_scores) + 1

# new_order = ['match_id','season','start_date','innings','over','ball','total_balls','balls_remaining','runs_from_ball','total_runs','team_score','runs_remaining','wickets_down', 'target_score','batting_team','bowling_team']
# df = df[new_order]


# df.to_csv('../data/modified_ODI_data.csv', index=False)

# print("done")



""" T20 dataset creation """


df = pd.read_csv('../data/ball_by_ball_it20.csv')

df = df.rename(columns = {'Innings':'innings', 'Ball':'ball','Runs From Ball':'runs_from_ball','Winner':'winner', 'Match ID':'match_id', 'Date':'start_date','Innings Runs':'team_score','Innings Wickets':'wickets_down','Bat First':'bat_first','Bat Second':'bat_second','Over':'over'})

df['total_runs'] = df.groupby(['match_id', 'innings'])['runs_from_ball'].transform('sum')

df['runs_remaining'] = df['total_runs'] - df['team_score']

df['start_date'] = pd.to_datetime(df['start_date'])

df['season'] = df['start_date'].dt.year

columns_to_drop = ['Unnamed: 0', 'Venue', 'Batter','Non Striker', 'Bowler','Batter Runs', 'Extra Runs', 'Ball Rebowled', 'Extra Type', 'Method','Player Out','Wicket','Runs to Get','Balls Remaining','Chased Successfully','Total Batter Runs','Total Non Striker Runs','Batter Balls Faced','Non Striker Balls Faced','Player Out Runs','Player Out Balls Faced','Bowler Runs Conceded','Valid Ball','Target Score']

df = df.drop(columns = columns_to_drop)


df['ball_no'] = df.groupby(['match_id', 'innings']).cumcount() + 1

df['total_balls'] = df.groupby(['match_id', 'innings'])['wickets_down'].transform('size')

df['balls_remaining'] = df['total_balls'] - df['ball_no']


inning1_scores = df[df['innings'] == 1].set_index('match_id')['total_runs'].to_dict()
df['target_score'] = -1 
df.loc[df['innings'] == 2, 'target_score'] = df['match_id'].map(inning1_scores) + 1


new_order = ['match_id','season','start_date','innings','over','ball','total_balls','balls_remaining','runs_from_ball','total_runs','team_score','runs_remaining','wickets_down','target_score','bat_first','bat_second','winner']

df = df[new_order]

df.to_csv('../data/modified_T20_data.csv', index=False)

print("done")

