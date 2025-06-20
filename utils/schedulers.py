import numpy as np

class exploration_rate_scheduler:
    """
    A class to manage the exploration rate (epsilon) for reinforcement learning agents.
    Supports both exponential and linear decay strategies.
    """
    
    def __init__(self, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=5*10**-4, policy='exp',**kwargs):
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.policy = policy
        
    def get_exploration_rate(self, episode, num_episodes):
        """
        Calculate the exploration rate based on the current episode and total episodes.
        
        Args:
            episode (int): Current episode number.
            num_episodes (int): Total number of episodes.
        
        Returns:
            float: The calculated exploration rate.
        """
        if self.policy == 'exp' or self.policy == 'exponential':
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-self.epsilon_decay * episode)
            epsilon = max(self.epsilon_end,epsilon)
            return epsilon
        
        elif self.policy == 'linear':
            decay = (self.epsilon_start - self.epsilon_end) / num_episodes
            epsilon = max(self.epsilon_end, self.epsilon_start - decay * episode)
            return epsilon
        
        else:
            raise ValueError("Unsupported exploration rate policy. Use 'exp' or 'linear'.")
        
    