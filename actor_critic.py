import torch
import torch.nn as nn
import torch.nn.functional as Functional
import numpy as np

HIDDEN_SIZE = 128

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # apply linear transformation y = xA^T + b
        # hidden layer 1, 18 X 128
        # handles raw input of actions and projects to HIDDEN_SIZE dimensions
        self.fc1 = nn.Linear(self.obs_dim, HIDDEN_SIZE)
        # hidden layer 2 128 x 64, compress actions
        self.fc2 = nn.Linear(HIDDEN_SIZE, 64)
        # Final action layer 64 x 5, compress again to the 5 actions an agent can take with probabilities
        self.final_action_layer = nn.Linear(64, self.act_dim)

    def forward(self, obs):
        '''
            Called when actor takes an action with ()

            Use ReLU to get logits of hiddent layers
            These logits are used in final action layer to 
            get probabilities of the five possibe actions
        '''
        obs = Functional.relu(self.fc1(obs))
        obs = Functional.relu(self.fc2(obs))
        logits = self.final_action_layer(obs)
        return logits
    
    def act(self, obs, epsilon=0.1):
        '''
            Determne agent action based on observations
            forward() call to determine our logits, logits used
            in softmax() call to get probabilities. Softmax()
            will normalize network output to a probability
            distribution.

            Agent action space: [no_action, move_left, move_right, move_down, move_up]
        '''
        logits = self.forward(obs)
        probs = Functional.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        # get logarithm of probability values
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate_actions(self, obs, actions):
        """
            Evaluate how good current policy is at choosing previously sampled actions

            Used in computation of log probability ratio in PPO
        """
        logits = self.forward(obs)
        probs = Functional.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        return log_probs
    
    def get_entropy(self, states):
        '''
            Encourage exploration by penalizing overly confident policies
                Higher Entropy -> policy is less certain, more exploration happened
                Lower Entropy -> policy is more certain, more exploitation happened
        
        
            For PPO, the goal should be to maximize entropy while optimizer
            will minimize total loss
        '''
        logits = self.forward(states)
        probs = Functional.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        entropy = dist.entropy()
        return entropy

class Critic(nn.Module):
    def __init__(self, obs_dim):
        super(Critic, self).__init__()
        self.obs_dim = obs_dim
        self.fc1 = nn.Linear(self.obs_dim, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, 64)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs):
        '''
            Returns a scalar value of expected future rewards
            Uses our 2 hidden layers: fc1 and fc2
            To be used in value loss and GAE.

            Implicitly called when we invoke critic (e.g. self.critic(states))
        '''
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # Automatically add batch dimension

        obs = torch.relu(self.fc1(obs))
        obs = torch.relu(self.fc2(obs))
        value = self.value_head(obs)

        return value.squeeze(-1)