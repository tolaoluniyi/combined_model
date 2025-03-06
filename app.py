from flask import Flask, render_template, request
import pandas as pd
from poisson_model import calculate_poisson_predictions
from bayesian_network import analyze_soccer_data
from markov_chain import MarkovChain
from monte_carlo import run_simulation
from dynamic_programming import DynamicProgramming

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/poisson', methods=['GET', 'POST'])
def poisson():
    result = None
    if request.method == 'POST':
        league = request.form['league']
        fixtures = pd.read_csv('data/fixtures.csv')
        tables = pd.read_csv('data/tables.csv')
        league_fixtures = fixtures[fixtures['league_name'] == league]
        league_tables = tables[tables['league_name'] == league]
        result = calculate_poisson_predictions(league_fixtures, league_tables)
    return render_template('poisson.html', result=result)

@app.route('/bayesian', methods=['GET', 'POST'])
def bayesian():
    result = None
    if request.method == 'POST':
        league = request.form['league']
        fixtures = pd.read_csv('data/fixtures.csv')
        tables = pd.read_csv('data/tables.csv')
        league_fixtures = fixtures[fixtures['league_name'] == league]
        league_tables = tables[tables['league_name'] == league]
        result = analyze_soccer_data(league_fixtures, league_tables)
    return render_template('bayesian.html', result=result)

@app.route('/markov', methods=['GET', 'POST'])
def markov():
    result = None
    if request.method == 'POST':
        league = request.form['league']
        fixtures = pd.read_csv('data/fixtures.csv')
        tables = pd.read_csv('data/tables.csv')
        league_fixtures = fixtures[fixtures['league_name'] == league]
        league_tables = tables[tables['league_name'] == league]
        markov_chain = MarkovChain(league_fixtures, league_tables)
        current_state = request.form['current_state']
        result = markov_chain.predict_next_state(current_state)
    return render_template('markov.html', result=result)

@app.route('/monte_carlo', methods=['GET', 'POST'])
def monte_carlo():
    result = None
    if request.method == 'POST':
        league = request.form['league']
        fixtures = pd.read_csv('data/fixtures.csv')
        tables = pd.read_csv('data/tables.csv')
        league_fixtures = fixtures[fixtures['league_name'] == league]
        league_tables = tables[tables['league_name'] == league]
        result = run_simulation(league_fixtures, league_tables)
    return render_template('monte_carlo.html', result=result)

@app.route('/dynamic_programming', methods=['GET', 'POST'])
def dynamic_programming():
    result = None
    if request.method == 'POST':
        league = request.form['league']
        fixtures = pd.read_csv('data/fixtures.csv')
        tables = pd.read_csv('data/tables.csv')
        league_fixtures = fixtures[fixtures['league_name'] == league]
        league_tables = tables[tables['league_name'] == league]
        dp_model = DynamicProgramming(league_fixtures, league_tables)
        team1 = request.form['team1']
        team2 = request.form['team2']
        result = dp_model.predict_outcome(team1, team2)
    return render_template('dynamic_programming.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
