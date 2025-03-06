import pandas as pd
import numpy as np

class MarkovChain:
    def __init__(self, fixtures, tables):
        self.fixtures = fixtures
        self.tables = tables
        self.states = self.get_states()
        self.transition_matrix = self.create_transition_matrix()

    def get_states(self):
        return list(set(self.tables['team']))

    def create_transition_matrix(self):
        n = len(self.states)
        matrix = np.zeros((n, n))
        state_index = {state: i for i, state in enumerate(self.states)}

        for index, row in self.fixtures.iterrows():
            team1 = row['team1']
            team2 = row['team2']
            matrix[state_index[team1], state_index[team2]] += 1

        matrix = matrix / matrix.sum(axis=1)[:, None]
        return matrix

    def predict_next_state(self, current_state):
        state_index = {state: i for i, state in enumerate(self.states)}
        current_index = state_index[current_state]
        next_state_index = np.argmax(self.transition_matrix[current_index])
        return self.states[next_state_index]
