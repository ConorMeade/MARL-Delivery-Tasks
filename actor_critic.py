import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

HIDDEN_SIZE = 128

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, output_dim)
        )

    def forward(self, obs):
        logits = self.net(obs)
        probs = F.softmax(logits, dim=-1)
        return probs

    def sample_action(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)

        probs = self.forward(obs)
        probs_np = probs.detach().cpu().numpy()

        action = np.random.choice(len(probs_np), p=probs_np)
        log_prob = np.log(probs_np[action] + 1e-8)  # stability epsilon

        return action, log_prob


class Critic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1)
        )

    def forward(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        return self.net(obs)
