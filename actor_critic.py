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
        self.fc1 = nn.Linear(self.obs_dim, HIDDEN_SIZE)
        # hidden layer 2 128 x 128
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        # Final action layer 128 x 5
        self.final_action_layer = nn.Linear(HIDDEN_SIZE, self.act_dim)

        # Shared fully connected layers
        # self.shared_fc = nn.Sequential(
        #     nn.Linear(obs_dim, HIDDEN_SIZE),
        #     nn.ReLU(),
        #     nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        #     nn.ReLU()
        # )
        
        # Actor network (outputs action probabilities)
        # self.actor_fc = nn.Linear(HIDDEN_SIZE, act_dim)
    def forward(self, obs):
        obs = Functional.relu(self.fc1(obs))
        obs = Functional.relu(self.fc2(obs))
        logits = self.final_action_layer(obs)
        return logits
    

    # def forward(self, obs):
    #     # Shared feature extraction
    #     x = self.shared_fc(obs)
        
    #     # Actor (action probabilities)
    #     logits = self.actor_fc(x)
        
    #     return logits

    def act(self, obs, epsilon=0.1):
        logits = self.forward(obs)
        probs = Functional.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        # if np.random.randn() < epsilon:
            # action = np.random.randint(probs.shape[-1])
            # log_prob = torch.log(probs.squeeze(0)[action] + 1e-10)
        # else:
        action = dist.sample()
        # get logarithm of probability values
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate_actions(self, obs, actions):
        """
        Compute log probability for given actions.
        """
        logits = self.forward(obs)
        probs = Functional.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        return log_probs
    
    def get_entropy(self, states):
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
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.value_head = nn.Linear(HIDDEN_SIZE, 1)
        # Shared fully connected layers
        # self.shared_fc = nn.Sequential(
        #     nn.Linear(obs_dim, HIDDEN_SIZE),
        #     nn.ReLU(),
        #     nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        #     nn.ReLU()
        # )
        
        # # Critic network (outputs state value)
        # self.critic_fc = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # Automatically add batch dimension

        obs = torch.relu(self.fc1(obs))
        obs = torch.relu(self.fc2(obs))
        value = self.value_head(obs)

        return value.squeeze(-1)