import json
import pandas as pd
import numpy as np
import urllib.request as ur
import requests
from datetime import timedelta, date
from sklearn.model_selection import train_test_split as tts

def authorize():
    path = 'APIauth.txt'
    temp = open(path,'r').read().splitlines()
    return (temp[0], temp[1])

def date_range(n):
    date_list = []

    for single_date in (date.today() - timedelta(n+1) for n in range(n)):
        date_list.append(single_date.strftime('%Y%m%d'))

    return(date_list)

def fetch_game_scoreboards():
    games_lists = []

    un, pw = authorize()

    url = 'https://api.mysportsfeeds.com/v1.1/pull/nhl/2017-2018-regular/scoreboard.json?fordate={date}'

    DATE_RANGE = 100
    n=0
    for i in date_range(DATE_RANGE):
        if n%50 == 0:
            print(n)
        n += 1
        try:
            json = requests.get(url.format(date=i), auth=(un, pw)).json()
            json['scoreboard']['gameScore']
        except:
            continue
        games_lists.append(json['scoreboard']['gameScore'])

    return(games_lists)

def fetch_team_data():
    un, pw = authorize()
    url = 'https://api.mysportsfeeds.com/v1.1/pull/nhl/2017-2018-regular/overall_team_standings.json'
    json = requests.get(url, auth=(un,pw)).json()
    data = json['overallteamstandings']['teamstandingsentry']
    return(data)

raw_game_data = fetch_game_scoreboards()
game_data = np.hstack(raw_game_data)
print(game_data.shape)
team_data = np.hstack(fetch_team_data())

print(team_data)

print(team_data[0])

from pandas.io.json import json_normalize

def fix_team_data():
    df = json_normalize(team_data)

    df = df[df.columns.drop(list(df.filter(regex='abbreviation|Abbreviation')))]

    df.drop(['team.Name', 'team.City'], axis=1, inplace=True)

    return(df)

team_standings = fix_team_data()


id_to_teamname = dict()

def parse_game(g):
    data = dict()
    away_team_id = g['game']['awayTeam']['ID']
    home_team_id = g['game']['homeTeam']['ID']
    data['away_team_id'] = away_team_id
    data['home_team_id'] = home_team_id
    if g['game']['awayTeam']['Abbreviation'] not in id_to_teamname:
        id_to_teamname[g['game']['awayTeam']['Abbreviation']] = g['game']['awayTeam']['ID']
    if g['game']['homeTeam']['Abbreviation'] not in id_to_teamname:
        id_to_teamname[g['game']['homeTeam']['Abbreviation']] = g['game']['homeTeam']['ID']
    if g['homeScore'] > g['awayScore']:
        data['winner'] = 1
    else:
        data['winner'] = 0
    return(pd.Series(data))

match_data = pd.DataFrame()

for g in game_data:
    match_data = match_data.append(parse_game(g), ignore_index=True)

print(match_data.shape)

match_data.drop('winner', axis=1).values

great_match_arrays = []

d = match_data.drop('winner', axis=1).values

for i in d:
    home = i[1]
    away = i[0]
    great_match_arrays.append(np.hstack([i, team_standings.loc[(team_standings['team.ID'] == home)].values.flatten(), team_standings.loc[(team_standings['team.ID'] == away)].values.flatten()]))
final_data = np.stack(great_match_arrays)

print(final_data.shape)

target = match_data['winner'].values
pd.DataFrame(target).to_csv('target.csv')

def count_rookies():
    game_ids = []
    for data in raw_game_data[0]:
        game_ids.append(data['game']['ID'])

    games = {}
    auth = authorize()
    lineup_url = 'https://api.mysportsfeeds.com/v1.1/pull/nhl/2017-2018-regular/game_startinglineup.json?gameid='
    players_json = requests.get('https://api.mysportsfeeds.com/v1.1/pull/nhl/2017-2018-regular/roster_players.json?fordate=20171025', auth=auth).json()

    for id in game_ids:
        game = []
        lineup_json = requests.get(lineup_url + id, auth=auth).json()
        home_rookies = 0
        away_rookies = 0
        for lineup_player in lineup_json['gamestartinglineup']['teamLineup'][0]['expected']['starter']:
            for player in players_json['rosterplayers']['playerentry']:
                if (player['player']['ID'] == lineup_player['player']['ID']) & (player['player']['IsRookie'] == 'true'):
                    home_rookies += 1

        for lineup_player in lineup_json['gamestartinglineup']['teamLineup'][1]['expected']['starter']:
            for player in players_json['rosterplayers']['playerentry']:
                if (player['player']['ID'] == lineup_player['player']['ID']) & (player['player']['IsRookie'] == 'true'):
                    away_rookies += 1

        game.append(home_rookies)
        game.append(away_rookies)
        games[id] = game
        return(games)
rookies = count_rookies()

target = match_data['winner'].values

X_train, X_test, y_train, y_test = tts(final_data, target, train_size=0.75, test_size=0.25, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

def predict_game(away, home):
    away_id = id_to_teamname[away]
    home_id = id_to_teamname[home]
    away_stats = team_standings.loc[(team_standings['team.ID'] == away_id)].values.flatten()
    home_stats = team_standings.loc[(team_standings['team.ID'] == home_id)].values.flatten()
    vec = np.hstack([away_id, home_id, home_stats, away_stats]).reshape(1, -1)

    res = pipeline.predict(vec.reshape(1, -1))
    if res == 1:
        return(home)
    else:
        return(away)

predict_game('BOS', 'CBJ')

team_data = np.hstack(fetch_team_data())

print(team_data.shape)

def multi_predict(m):
    ret = []
    for i in m:
        ret.append(predict_game(i[0], i[1]))
    return(ret)

predictions = [['SJS', 'BOS'], ['CAR', 'TOR'], ['ARI', 'NYR'], ['WPJ', 'PIT'], ['LAK', 'MTL'], ['PHI', 'OTT'], ['DET', 'TBL'], ['ANA', 'FLO'], ['NYI', 'MIN'], ['DAL', 'EDM'], ['WSH', 'VAN']]

print(multi_predict(predictions))

print(id_to_teamname)
