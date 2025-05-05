import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from actor_critic import Actor, Critic

class MAPPO:
    def __init__(self, env, actor_critic, lr=1e-4, gamma=0.95, gae_lambda=0.90, clip_epsilon=0.15, value_loss_coef=0.5, entropy_coef=0.01):
        self.env = env
        self.actor = Actor(actor_critic.obs_dim, actor_critic.act_dim)
        self.critic = Critic(actor_critic.obs_dim)
        # Adam gradient descent
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=0.001, weight_decay=0.00001)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

    # def compute_advantages(self, rewards, values, next_values, dones):
    #     # compare how well an action is compared to average action using generalize advantage estimation (GAE)
    #     # reduce variance of policy gradient estimates
    #     advantages = torch.zeros_like(rewards)
    #     last_advantage = 0

    #     for t in reversed(range(len(rewards))):
    #         if dones[t]:
    #             delta = rewards[t] - values[t]
    #         else:
    #             # compute TD error
    #             delta = rewards[t] + self.gamma * next_values[t] - values[t]
    #         last_advantage = delta + self.gamma * self.gae_lambda * last_advantage
    #         advantages[t] = last_advantage
    #     return advantages
    
    def compute_advantages(self, rewards, values, next_values, dones):
        advantages = torch.zeros_like(rewards)
        last_advantage = 0

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]  # 0 if done, 1 otherwise
            delta = rewards[t] + self.gamma * next_values[t] * mask - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * mask * last_advantage

        returns = advantages + values
        return advantages, returns
        
    #  [env step] ➔ [save rollout] ➔ [finish batch] ➔
    #     ➔ [compute advantage] ➔ [compute losses] ➔ [backprop] ➔ [update networks]
    def update_mappo(self, rollouts, next_obs):
        '''
            Convert rollouts we get from training loops into tensors for states, action, log_prob,
            rewards, next_states, and terminations.

            Use critic object to get values of current states and values from next_states.

            Call compute_advantages() to generate A_t in MAPPO. This will allow use to compare the value
            of an action to how good it is compared to the average action
        
        '''
        # print('Updating Policy...')
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
        rewards = torch.stack(
            [torch.tensor(r['reward'], dtype=torch.float32) for r in rollouts]
        )
        next_states = torch.stack([torch.tensor(r['next_state'], dtype=torch.float32) for r in rollouts])
        # next_obs = torch.stack(
        #     [torch.tensor(r['next_value'], dtype=torch.float32) for r in rollouts]
        # )
        # advantage is the diff between Q-value and the value function:
        # advantages = torch.stack([r['advantage'] for r in rollouts])
        terminations = torch.stack([torch.tensor(r['termination'], dtype=torch.float32) for r in rollouts])

        n_vals = torch.stack([torch.tensor(r['next_value'], dtype=torch.float32) for r in rollouts])

        # Get values (both current and next) from the critic
        values = self.critic(states)
        next_values = self.critic(next_states)
        # next_values = torch.tensor([r['next_value'] for r in rollouts], dtype=torch.float32)
        # compute bootstrapped returns/advantages
        # next_values = self.critic(next_states)   
        # next_values = torch.roll(values, shifts=-1, dims=0)  # Using next value from the subsequent timestep

        # print("Mean value:", values.mean().item())
        # print("Mean next_value:", next_values.mean().item())
        # if values != next_values:
            # print('vals change')

        advantages, returns = self.compute_advantages(rewards, values, next_values, terminations)
        # advantages = torch.stack(
        #     [torch.tensor(r['advantage'], dtype=torch.float32) for r in rollouts]
        # )

        # Normalize advantages
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # Get new log probs and values
        # new_log_probs = self.actor.evaluate_actions(next_states, actions)
        new_log_probs = self.actor.evaluate_actions(states, actions)
        # new_values = self.critic(states)

        # Determine ratio of new and old log probs, used in PPO
        ratio = torch.exp(new_log_probs - old_log_probs)

        # L^CLIP(θ) = E_t [ min( r_t(θ) * A_t, clip(r_t(θ), 1 - ε, 1 + ε) * A_t ) ]
        # Policy loss (clipped PPO objective)
        # Avoid making large updates to a policy through clipping the probability ratio between new and old
        # policies which allows learning to stabalize
        policy_loss = torch.min(ratio * advantages, torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages).mean()

        # Compute the value loss, mse (c_v)
        value_loss = torch.nn.functional.mse_loss(values, returns)
        # value_loss = self.value_loss_coef * (returns - next_values).pow(2).mean()
        
        # average uncertainty over all possible actions with entropy, higher entropy values indicate more explorationx
        entropy = self.actor.get_entropy(states)
        
        # Compute the entropy loss, to maximize entropy, we subtract it from the total loss (loss var here)
        entropy_loss = -self.entropy_coef * entropy.mean()

        # Total loss, entropy regularization
        # L_total = L_Policy + c_v * L_value_loss * policy entropy
        l_loss = policy_loss + value_loss + entropy_loss

        # Total loss
        # loss = policy_loss + self.value_loss_coef * value_loss - (self.entropy_coef * entropy)

        # Backpropagation
        self.optimizer.zero_grad()
        l_loss.backward()
        self.optimizer.step()

