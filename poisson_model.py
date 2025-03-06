import pandas as pd
import numpy as np
from scipy.stats import poisson

def calculate_poisson_predictions(fixtures, tables):
    tables['home_goals_avg'] = tables['home_goals_scored'] / tables['home_matches_played']
    tables['away_goals_avg'] = tables['away_goals_scored'] / tables['away_matches_played']
    
    predictions = []
    for index, row in fixtures.iterrows():
        team1 = row['team1']
        team2 = row['team2']
        team1_stats = tables[tables['team'] == team1].iloc
        team2_stats = tables[tables['team'] == team2].iloc
        
        home_goals = poisson.pmf(k=np.arange(0, 6), mu=team1_stats['home_goals_avg'])
        away_goals = poisson.pmf(k=np.arange(0, 6), mu=team2_stats['away_goals_avg'])
        
        outcome = 'Draw'
        if home_goals.mean() > away_goals.mean():
            outcome = 'Home Win'
        elif away_goals.mean() > home_goals.mean():
            outcome = 'Away Win'
        
        over1_5 = (home_goals[2:].sum() + away_goals[2:].sum()) > 1.5
        over2_5 = (home_goals[3:].sum() + away_goals[3:].sum()) > 2.5
        
        predictions.append({
            'match': f"{team1} vs {team2}",
            'outcome': outcome,
            'over1.5': over1_5,
            'over2.5': over2_5
        })
    
    return predictions
