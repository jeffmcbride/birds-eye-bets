from datetime import datetime
from datetime import timedelta
import pandas as pd
from sportsipy.nba.boxscore import Boxscores, Boxscore
from sportsipy.nba.teams import Teams
import numpy as np
import sklearn as sk
from sklearn import *
from scipy import stats


# Method to drop prefix from every column in a df    
def replace_prefix(df, prefix, replacement):
    df.columns = df.columns.str.replace(prefix, replacement)
    return df
    
''' 
Given: is a dataframe of games and a dataframe of stats for those games
Output: Dataframes for all away team stats, and all home team stats
'''
def game_data(game_df,game_stats):
    try:
        away_team_df = game_df[['away_abbr', 'away_score']].rename(columns = {'away_abbr': 'team_abbr', 'away_score': 'score'})
        away_team_df['h/a'] = 'a'
        away_team_df['opp'] = game_df['home_abbr']
        home_team_df = game_df[['home_abbr', 'home_score']].rename(columns = {'home_abbr': 'team_abbr', 'home_score': 'score'})
        home_team_df['h/a'] = 'h'
        home_team_df['opp'] = game_df['away_abbr']
        try:
            if game_df.loc[0,'away_score'] > game_df.loc[0,'home_score']:
                away_team_df = pd.merge(away_team_df, pd.DataFrame({'game_won' : [1], 'game_lost' : [0]}),left_index = True, right_index = True)
                home_team_df = pd.merge(home_team_df, pd.DataFrame({'game_won' : [0], 'game_lost' : [1]}),left_index = True, right_index = True)
            elif game_df.loc[0,'away_score'] < game_df.loc[0,'home_score']:
                away_team_df = pd.merge(away_team_df, pd.DataFrame({'game_won' : [0], 'game_lost' : [1]}),left_index = True, right_index = True)
                home_team_df = pd.merge(home_team_df, pd.DataFrame({'game_won' : [1], 'game_lost' : [0]}),left_index = True, right_index = True)
            else: 
                away_team_df = pd.merge(away_team_df, pd.DataFrame({'game_won' : [0], 'game_lost' : [0]}),left_index = True, right_index = True)
                home_team_df = pd.merge(home_team_df, pd.DataFrame({'game_won' : [0], 'game_lost' : [0]}),left_index = True, right_index = True)
        except TypeError:
                away_team_df = pd.merge(away_team_df, pd.DataFrame({'game_won' : [np.nan], 'game_lost' : [np.nan]}),left_index = True, right_index = True)
                home_team_df = pd.merge(home_team_df, pd.DataFrame({'game_won' : [np.nan], 'game_lost' : [np.nan]}),left_index = True, right_index = True)     
            
        # List of stats to grab
        away_stats_df = game_stats.dataframe[['away_assist_percentage',
                                               'away_assists',
                                               'away_block_percentage',
                                               'away_blocks',
                                               'away_defensive_rating',
                                               'away_defensive_rebound_percentage',
                                               'away_defensive_rebounds',
                                               'away_effective_field_goal_percentage',
                                               'away_field_goal_percentage',
                                               'away_field_goals',
                                               'away_free_throw_attempt_rate',
                                               'away_free_throw_percentage',
                                               'away_free_throws',
                                               'away_personal_fouls',
                                               'away_steal_percentage',
                                               'away_steals',
                                               'away_three_point_attempt_rate',
                                               'away_three_point_field_goal_percentage',
                                               'away_three_point_field_goals',
                                               'away_total_rebound_percentage',
                                               'away_total_rebounds',
                                               'away_true_shooting_percentage',
                                               'away_turnover_percentage',
                                               'away_turnovers',
                                               'away_two_point_field_goal_percentage',
                                               'away_two_point_field_goals',
                                               'home_defensive_rebounds',
                                               'home_effective_field_goal_percentage',
                                               'home_field_goal_percentage',
                                               'home_field_goals',
                                               'home_personal_fouls',
                                               'home_three_point_field_goal_percentage',
                                               'home_three_point_field_goals',
                                               'home_total_rebounds',
                                               'home_true_shooting_percentage',
                                               'home_two_point_field_goal_percentage',
                                               'home_two_point_field_goals',
                                               'date'
                                               ]].reset_index().drop(columns ='index')
        home_stats_df = game_stats.dataframe[['home_assist_percentage',
                                               'home_assists',
                                               'home_block_percentage',
                                               'home_blocks',
                                               'home_defensive_rating',
                                               'home_defensive_rebound_percentage',
                                               'home_defensive_rebounds',
                                               'home_effective_field_goal_percentage',
                                               'home_field_goal_percentage',
                                               'home_field_goals',
                                               'home_free_throw_attempt_rate',
                                               'home_free_throw_percentage',
                                               'home_free_throws',
                                               'home_personal_fouls',
                                               'home_steal_percentage',
                                               'home_steals',
                                               'home_three_point_attempt_rate',
                                               'home_three_point_field_goal_percentage',
                                               'home_three_point_field_goals',
                                               'home_total_rebound_percentage',
                                               'home_total_rebounds',
                                               'home_true_shooting_percentage',
                                               'home_turnover_percentage',
                                               'home_turnovers',
                                               'home_two_point_field_goal_percentage',
                                               'home_two_point_field_goals',
                                               'away_defensive_rebounds',
                                               'away_effective_field_goal_percentage',
                                               'away_field_goal_percentage',
                                               'away_field_goals',
                                               'away_personal_fouls',
                                               'away_three_point_field_goal_percentage',
                                               'away_three_point_field_goals',
                                               'away_total_rebounds',
                                               'away_true_shooting_percentage',
                                               'away_two_point_field_goal_percentage',
                                               'away_two_point_field_goals',
                                               'date'
                                               ]].reset_index().drop(columns ='index')       
        
        away_stats_df = replace_prefix(away_stats_df, 'away_', '')
        away_stats_df = replace_prefix(away_stats_df, 'home_', 'opp_')
        home_stats_df = replace_prefix(home_stats_df, 'home_', '')
        home_stats_df = replace_prefix(home_stats_df, 'away_', 'opp_')

        away_team_df = pd.merge(away_team_df, away_stats_df,left_index = True, right_index = True)
        home_team_df = pd.merge(home_team_df, home_stats_df,left_index = True, right_index = True)
    except TypeError:
        away_team_df = pd.DataFrame()
        home_team_df = pd.DataFrame()
    return away_team_df, home_team_df



def get_day_stats(date, games_dict):
    df = pd.DataFrame()
    for i in range(len(games_dict[date])):
        stats = Boxscore(games_dict[date][i]['boxscore'])
        game_df = pd.DataFrame(games_dict[date][i], index = [0])
        away_team, home_team = game_data(game_df, stats)
        away_team['date'] = pd.to_datetime(away_team['date'], format='%I:%M %p, %B %d, %Y')
        home_team['date'] = pd.to_datetime(home_team['date'], format='%I:%M %p, %B %d, %Y')
        df = pd.concat([df, away_team])
        df = pd.concat([df, home_team])    
    return df
    

def get_schedule(date):
    teams = Teams()
    df_full = pd.DataFrame()
    for team in teams:
        schedule = team.schedule
        df = team.schedule.dataframe
        df['team_abbr'] = team.abbreviation
        df_full = pd.concat([df_full, df])
        df_full = df_full[df_full['location'] == 'Away']
    return df_full[df_full['datetime'].dt.date == date.date()]


def get_season_stats(start_date, end_date):
    games = Boxscores(start_date, end_date)
    games_dict = games.games
    delta = timedelta(days=1)
    total_df = pd.DataFrame()
    while start_date <= end_date:
        print(start_date)
        date = start_date.strftime('%#m-%#d-%Y')
        df = get_day_stats(date, games_dict)
        total_df = pd.concat([total_df,df])
        start_date += delta
    return total_df

def agg_data_to_date(games, start_date, end_date):
    agg_games_df = pd.DataFrame()
    delta = timedelta(days=1)
    all_games_df = games[['h/a', 'team_abbr', 'date', 'opp']]
    all_games_df['date'] = all_games_df['date'].dt.date
    while start_date <= end_date:
        print(start_date)
        games_df = all_games_df[all_games_df['date'] == start_date.date()]
        agg_weekly_df = games[games.date < start_date].drop(columns = ['game_won', 'game_lost', 'h/a']).groupby(by=["team_abbr"]).mean().reset_index()
        agg_weekly_df = agg_weekly_df.rename(columns = {'score': 'ppg'})
        win_loss_df = games[games.date < start_date][["team_abbr",'game_won', 'game_lost']].groupby(by=["team_abbr"]).sum().reset_index()
        win_loss_df['win_perc'] = win_loss_df['game_won'] / (win_loss_df['game_won'] + win_loss_df['game_lost'])
        win_loss_df['GP'] = win_loss_df['game_won'] + win_loss_df['game_lost']
        
        score = games[games.date.dt.date == start_date.date()][["team_abbr", "score"]]
        last_game = games[games['date'] < start_date].groupby('team_abbr').agg({'date':'max'})
        last_game = last_game.rename(columns = {"date": "last_game"} )
        
        
        
        agg_weekly_df = pd.merge(win_loss_df,agg_weekly_df,left_on = ['team_abbr'], right_on = ['team_abbr'])
        
        agg_weekly_df = pd.merge(score,agg_weekly_df,how = 'outer', left_on = ['team_abbr'], right_on = ['team_abbr'])
        agg_weekly_df = pd.merge(last_game,agg_weekly_df,how = 'outer', left_on = ['team_abbr'], right_on = ['team_abbr'])
        
        agg_weekly_df = agg_weekly_df[agg_weekly_df['GP'] >= 1]
        away_df = pd.merge(games_df,agg_weekly_df,how = 'inner', left_on = ['team_abbr'], right_on = ['team_abbr'])

        away_df = away_df[away_df['h/a'] == 'a']
        away_df['days_between_games'] = (away_df['date'] - away_df['last_game'].dt.date)

        away_df = away_df.drop(columns = 'last_game')
        away_df = away_df.drop(columns = 'h/a')
        away_df = away_df.add_prefix('away_')

        
        home_df = pd.merge(games_df,agg_weekly_df,how = 'inner', left_on = ['team_abbr'], right_on = ['team_abbr'])
        home_df['days_between_games'] = (home_df['date'] - home_df['last_game'].dt.date)

        home_df = home_df.drop(columns = 'last_game')

        home_df = home_df[home_df['h/a'] == 'h']
        home_df = home_df.drop(columns = 'h/a')
        home_df = home_df.add_prefix('home_')
     
        
        agg_weekly_df = pd.merge(away_df,home_df,left_on = ['away_team_abbr', 'away_opp'], right_on = ['home_opp', 'home_team_abbr'])    
        print(agg_weekly_df)
        agg_weekly_df = agg_weekly_df.drop(columns = ['away_date', 'away_opp', 'home_opp']).rename(columns = {'home_date': 'date'})
        agg_weekly_df = agg_weekly_df[sorted(agg_weekly_df.columns, key=lambda x: (x[5:]))]
        cols_to_move = ['date', 'home_team_abbr', 'home_score', 'away_team_abbr', 'away_score']
        agg_weekly_df = agg_weekly_df[cols_to_move + [col for col in agg_weekly_df.columns if col not in cols_to_move]]
        agg_games_df = pd.concat([agg_games_df, agg_weekly_df])
        start_date += delta
    agg_games_df = agg_games_df.reset_index().drop(columns = 'index')
    agg_games_df['total_score'] = agg_games_df['away_score'] + agg_games_df['home_score']
    return agg_games_df
    

def display(results, df, attribute):
    df = df.reset_index().drop(columns = 'index')
    column = attribute + ' prediction'
    df[column] = None
    print(results)
    for j in range(len(results)):
        prediction = results[j]
        df.loc[j, column] = prediction
    return df


def fit_test(attribute, X_train, y_train, X_test, y_test):
    results = []
    classifiers = [
    sk.svm.SVR(),
    sk.linear_model.LinearRegression(),
    sk.linear_model.BayesianRidge(),
    sk.linear_model.ARDRegression(),
    sk.linear_model.TheilSenRegressor(),
    sk.linear_model.Ridge(),
    sk.linear_model.Lasso(),
    sk.linear_model.ElasticNet()
    ]
    for item in classifiers:
        clf = item
        clf.fit(X_train, y_train[attribute])
        pred = clf.predict(X_test)
        results.append(pred)
    return stats.trim_mean(results, 0.4).tolist()



def read_from_csv(file):
    date_cols = ['date']
    stats_df = pd.read_csv(file, parse_dates=date_cols)
    stats_df = stats_df.drop(columns = {'Unnamed: 0'})
    return stats_df
    
