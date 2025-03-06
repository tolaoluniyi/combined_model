import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

def analyze_soccer_data(fixtures, tables):
    # Calculate average goals
    tables['home_goals_avg'] = tables['home_goals_scored'] / tables['home_matches_played']
    tables['away_goals_avg'] = tables['away_goals_scored'] / tables['away_matches_played']
    
    # Define the structure of the Bayesian Network
    model = BayesianNetwork([('home_goals_avg', 'outcome'),
                             ('away_goals_avg', 'outcome'),
                             ('home_goals_conceded', 'outcome'),
                             ('away_goals_conceded', 'outcome')])
    
    # Fit the model using Maximum Likelihood Estimation
    model.fit(tables, estimator=MaximumLikelihoodEstimator)
    
    predictions = []
    for index, row in fixtures.iterrows():
        team1 = row['team1']
        team2 = row['team2']
        team1_stats = tables[tables['team'] == team1].iloc
        team2_stats = tables[tables['team'] == team2].iloc
        
        outcome = model.predict(pd.DataFrame([{
            'home_goals_avg': team1_stats['home_goals_avg'],
            'away_goals_avg': team2_stats['away_goals_avg'],
            'home_goals_conceded': team1_stats['home_goals_conceded'],
            'away_goals_conceded': team2_stats['away_goals_conceded']
        }]))
        
        over1_5 = (team1_stats['home_goals_avg'] + team2_stats['away_goals_avg']) > 1.5
        over2_5 = (team1_stats['home_goals_avg'] + team2_stats['away_goals_avg']) > 2.5
        
        predictions.append({
            'match': f"{team1} vs {team2}",
            'outcome': outcome,
            'over1.5': over1_5,
            'over2.5': over2_5
        })
    
    return predictions
