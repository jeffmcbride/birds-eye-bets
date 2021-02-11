import nba
from scipy import stats
from datetime import timedelta
from datetime import datetime
import pandas as pd
import numpy as np


start_date = datetime(2020, 12, 22)
end_date = datetime.now()
yesterday = datetime.now() - timedelta(days=1)

last_season_start_date = datetime(2019, 10, 22)
last_season_end_date = datetime(2020, 3, 11)

# last_season = nba.get_season_stats(last_season_start_date, last_season_end_date)
# last_season_agg =  nba.agg_data_to_date(last_season, last_season_start_date, last_season_end_date)
# last_season_agg['away_days_between_games'] = last_season_agg['away_days_between_games'].dt.days
# last_season_agg['home_days_between_games'] = last_season_agg['home_days_between_games'].dt.days
# last_season_agg = nba.read_from_csv('last_season.csv')
# stats_df = nba.read_from_csv('stats_df.csv')

stats_df_since_last = nba.get_season_stats(yesterday, end_date)

stats_df = pd.concat([stats_df, stats_df_since_last])





schedule = nba.get_schedule(end_date)

df_away = pd.DataFrame(columns=stats_df.columns)
df_away['team_abbr'] = schedule['team_abbr']
df_away['opp'] = schedule['opponent_abbr']
df_away['h/a'] = 'a'
df_away['date'] = schedule['datetime']

df_home = pd.DataFrame(columns=stats_df.columns)
df_home['team_abbr'] = schedule['opponent_abbr']
df_home['opp'] = schedule['team_abbr']
df_home['h/a'] = 'h'
df_home['date'] = schedule['datetime']

df_to_date = pd.concat([stats_df,df_away,df_home])

df_to_date = df_to_date.reset_index(drop=True)
df_to_date = df_to_date.fillna(0)

aggregated_df = nba.agg_data_to_date(df_to_date, start_date, end_date)


aggregated_df['away_days_between_games'] = aggregated_df['away_days_between_games'].dt.days
aggregated_df['home_days_between_games'] = aggregated_df['home_days_between_games'].dt.days
try:
    aggregated_df = aggregated_df.drop(columns = [
            'away_free_throw_percentage',
           'home_free_throw_percentage',
           'away_opp_three_point_field_goal_percentage',
           'home_opp_three_point_field_goal_percentage',
           'away_three_point_field_goal_percentage',
           'home_three_point_field_goal_percentage'])
except: 
    aggregated_df = aggregated_df.drop(columns = [
       'away_opp_three_point_field_goal_percentage',
       'home_opp_three_point_field_goal_percentage',
       'away_three_point_field_goal_percentage',
       'home_three_point_field_goal_percentage'])
aggregated_df = pd.concat([last_season_agg, aggregated_df])

train_df = aggregated_df[aggregated_df['date'] < end_date.date()]
test_df = aggregated_df[aggregated_df['date'] == end_date.date()]

# msk = np.random.rand(len(aggregated_df)) < 0.8
# train_df = aggregated_df[msk]
# test_df = aggregated_df[~msk]

test_results_df = test_df[['date','away_team_abbr', 'away_score','home_team_abbr','home_score', 'total_score']].reset_index()

X_train = train_df.drop(columns = ['away_team_abbr', 'home_team_abbr', 'date','away_score', 'home_score', 'total_score'])
y_train = train_df[['away_score', 'home_score', 'total_score']]
X_test = test_df.drop(columns = ['away_team_abbr', 'home_team_abbr', 'date','away_score', 'home_score', 'total_score'])
y_test = test_df[['away_score', 'home_score', 'total_score']]



attributes = ['away_score', 'home_score', 'total_score']





away = nba.fit_test('away_score', X_train, y_train, X_test, y_test)
home = nba.fit_test('home_score', X_train, y_train, X_test, y_test)
total = nba.fit_test('total_score', X_train, y_train, X_test, y_test)
df = nba.display(away,test_results_df, 'away_score')
df = nba.display(home,df,'home_score')
df = nba.display(total,df,'total_score')


df.to_html('02-10-history.html')