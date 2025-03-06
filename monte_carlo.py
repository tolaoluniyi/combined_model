import pandas as pd
import numpy as np

def simulate_match(home_goals_avg, away_goals_avg, num_simulations=10000):
    home_wins = 0
    away_wins = 0
    draws = 0
    over1_5 = 0
    over2_5 = 0

    for _ in range(num_simulations):
        home_goals = np.random.poisson(home_goals_avg)
        away_goals = np.random.poisson(away_goals_avg)

        if home_goals > away_goals:
            home_wins += 1
        elif away_goals > home_goals:
            away_wins += 1
        else:
            draws += 1

        if home_goals + away_goals > 1.5:
            over1_5 += 1
        if home_goals + away_goals > 2.5:
            over2_5 += 1

    return home_wins, away_wins, draws, over1_5, over2_5

def run_simulation(fixtures, tables):
    results = []

    for index, row in fixtures.iterrows():
        team1 = row['team1']
        team2 = row['team2']
        team1_stats = tables[tables['team'] == team1].iloc
        team2_stats = tables[tables['team'] == team2].iloc

        home_wins, away_wins, draws, over1_5, over2_5 = simulate_match(team1_stats['home_goals_avg'], team2_stats['away_goals_avg'])
        
        results.append({
            'match': f"{team1} vs {team2}",
            'outcome': 'Home Win' if home_wins > away_wins else 'Away Win' if away_wins > home_wins else 'Draw',
            'over1.5': over1_5 / num_simulations,
            'over2.5': over2_5 / num_simulations
        })
