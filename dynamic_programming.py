import pandas as pd
import numpy as np

class DynamicProgramming:
    def __init__(self, fixtures, tables):
        self.fixtures = fixtures
        self.tables = tables
        self.teams = self.get_teams()
        self.dp_table = self.create_dp_table()

    def get_teams(self):
        return list(set(self.tables['team']))

    def create_dp_table(self):
        n = len(self.teams)
        table = np.zeros((n, n))
        team_index = {team: i for i, team in enumerate(self.teams)}

        for index, row in self.fixtures.iterrows():
            team1 = row['team1']
            team2 = row['team2']
            team1_stats = self.tables[self.tables['team'] == team1].iloc
            team2_stats = self.tables[self.tables['team'] == team2].iloc
            
            table[team_index[team1], team_index[team2]] = team1_stats['home_goals_avg']
            table[team_index[team2], team_index[team1]] = team2_stats['away_goals_avg']

        return table

    def predict_outcome(self, team1, team2):
        team_index = {team: i for i, team in enumerate(self.teams)}
        index1 = team_index[team1]
        index2 = team_index[team2]
        
        home_goals_avg = self.dp_table[index1, index2]
        away_goals_avg = self.dp_table[index2, index1]
        
        outcome = 'Draw'
        if home_goals_avg > away_goals_avg:
            outcome = 'Home Win'
        elif away_goals_avg > home_goals_avg:
            outcome = 'Away Win'
        
        over1_5 = home_goals_avg + away_goals_avg > 1.5
        over2_5 = home_goals_avg + away_goals_avg > 2.5
        
        return outcome, over1_5, over2_5
