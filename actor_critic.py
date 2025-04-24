import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

HIDDEN_SIZE = 128

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        # Shared fully connected layers
        self.shared_fc = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU()
        )
        
        # Actor network (outputs action probabilities)
        self.actor_fc = nn.Linear(HIDDEN_SIZE, act_dim)

    def forward(self, obs):
        # Shared feature extraction
        x = self.shared_fc(obs)
        
        # Actor (action probabilities)
        logits = self.actor_fc(x)
        
        return logits

    def act(self, obs):
        """
        Sample action based on the policy (Actor).
        """
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)  # Discrete action space
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob

    def evaluate_actions(self, obs, actions):
        """
        Compute log probability for given actions.
        """
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        return log_probs

class Critic(nn.Module):
    def __init__(self, obs_dim):
        super(Critic, self).__init__()
        self.obs_dim = obs_dim
        
        # Shared fully connected layers
        self.shared_fc = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU()
        )
        
        # Critic network (outputs state value)
        self.critic_fc = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, obs):
        # Shared feature extraction
        x = self.shared_fc(obs)
        
        # Critic (state value)
        value = self.critic_fc(x)
        
        return value