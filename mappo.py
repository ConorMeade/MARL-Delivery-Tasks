import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from actor_critic import Actor, Critic

class MAPPO:
    def __init__(self, env, actor_critic, lr=1e-4, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.15, value_loss_coef=0.5, entropy_coef=0.01):
        self.env = env
        self.actor = Actor(actor_critic.obs_dim, actor_critic.act_dim)  # Instantiate Actor
        self.critic = Critic(actor_critic.obs_dim)  # Instantiate Critic
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=0.001, weight_decay=0.00001)

        # Hyperparams
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

    def compute_advantages(self, rewards, values, next_values, dones):
        # compare how well an action is compared to average action using generalize advantage estimation (GAE)
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
            else:
                delta = rewards[t] + self.gamma * next_values[t] - values[t]
            last_advantage = delta + self.gamma * self.gae_lambda * last_advantage
            advantages[t] = last_advantage
        return advantages
    
    #  [env step] ➔ [save rollout] ➔ [finish batch] ➔
    #     ➔ [compute advantage] ➔ [compute losses] ➔ [backprop] ➔ [update networks]
    def update_mappo(self, rollouts):
        # convert rollouts into tensors
        states = torch.stack([
            torch.tensor(r['state'], dtype=torch.float32) for r in rollouts
        ])
        # actions = torch.stack([r['action'] for r in rollouts])
        actions = torch.stack([
            torch.tensor(r['action'], dtype=torch.float32) for r in rollouts
        ])
        # old_log_probs = torch.stack([r['log_prob'] for r in rollouts])
        old_log_probs = torch.stack(
            [torch.tensor(r['log_prob'], dtype=torch.float32) for r in rollouts]
        )
        # returns = torch.stack([r['return'] for r in rollouts])
        returns = torch.stack(
            [torch.tensor(r['reward'], dtype=torch.float32) for r in rollouts]
        )
        # advantage is the diff between Q-value and the value function:
        # advantages = torch.stack([r['advantage'] for r in rollouts])
        terminations = torch.stack([torch.tensor(r['termination'], dtype=torch.float32) for r in rollouts])


        # Get values (both current and next) from the critic
        values = self.critic(states)
        next_values = torch.roll(values, shifts=-1, dims=0)  # Using next value from the subsequent timestep


        advantages = self.compute_advantages(returns, values, next_values, terminations)
        # advantages = torch.stack(
        #     [torch.tensor(r['advantage'], dtype=torch.float32) for r in rollouts]
        # )

        # Get new log probs and values
        new_log_probs = self.actor.evaluate_actions(states, actions)
        new_values = self.critic(states)
        entropy = self.actor.get_entropy(states)

        # Compute the ratio of new and old log probs
        ratio = torch.exp(new_log_probs - old_log_probs)

        # L^CLIP(θ) = E_t [ min( r_t(θ) * A_t, clip(r_t(θ), 1 - ε, 1 + ε) * A_t ) ]
        # Compute the policy loss (clipped PPO objective)
        policy_loss = torch.min(ratio * advantages, torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages).mean()

        # Compute the value loss
        value_loss = self.value_loss_coef * (returns - new_values).pow(2).mean()

        # Compute the entropy loss
        entropy_loss = -self.entropy_coef * entropy.mean()

        # Total loss
        loss = policy_loss + value_loss + entropy_loss

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

