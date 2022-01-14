import numpy as np


class PPOController:
    def __init__(self, init_state, model, padding=32):
        self.init_state = init_state
        self.state = init_state
        self.model = model
        self.padded_obs = np.zeros((padding, 1))

    def predict(self, observation):
        self.padded_obs[0] = observation
        self.state = observation
        action, _ = self.model.predict(self.padded_obs, deterministic=True)
        return {'basal_rate':action[0][0]}

    def reset(self):
        self.state = self.init_state
