import numpy as np
from gym import Wrapper
import torch

class NanTerminationWrapper(Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        
        # Check for NaN values in the observation
        nan_mask = torch.isnan(observation)
        if nan_mask.any():
            print("Nan detected in observation", observation)
            done = True  # Terminate the environment
            observation = self.reset()  # Reset the environment

        return observation, reward, done, info 

    def reset(self):
        observation = self.env.reset()
        """
        if isinstance(observation, dict):
            observation = {k: torch.tensor(v) for k, v in observation.items()}  # Ensure each value is a tensor
        """
        return observation