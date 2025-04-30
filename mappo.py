import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from actor_critic import Actor, Critic

class MAPPO:
    def __init__(self, env, actor_critic, lr=1e-4, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2, value_loss_coef=0.5, entropy_coef=0.01):
        self.env = env
        self.actor = Actor(actor_critic.obs_dim, actor_critic.act_dim)  # Instantiate Actor
        self.critic = Critic(actor_critic.obs_dim)  # Instantiate Critic
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=0.001)

        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

    def compute_advantages(self, rewards, values, next_values, dones):
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
            else:
                delta = rewards[t] + self.gamma * next_values[t] - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * last_advantage
        return advantages
    


    #  [env step] ➔ [save rollout] ➔ [finish batch] ➔
    #     ➔ [compute advantage] ➔ [compute losses] ➔ [backprop] ➔ [update networks]
    def update_mappo(self, rollouts):
        # Unroll the stored rollouts into tensors
        # states = torch.stack([r['state'] for r in rollouts])
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


    def train(self, num_episodes=1000):
        # generate rollouts to train on
        for _ in range(num_episodes):
            rollouts = []
            obs = self.env.reset()
            done = False
            while not done:
                actions = {}
                log_probs = {}
                for agent in self.env.agents:
                    action, log_prob = self.actor.act(obs[agent])
                    actions[agent] = action
                    log_probs[agent] = log_prob

                # Step environment with actions
                next_obs, rewards, dones, truncs, infos = self.env.step(actions)

                next_obs_array = next_obs[agent]

                if isinstance(next_obs_array, np.ndarray):
                    next_obs_tensor = torch.tensor(next_obs_array, dtype=torch.float32)

                if next_obs_tensor.dim() == 1:
                    next_obs_tensor = next_obs_tensor.unsqueeze(0)  # Add batch dim

                next_value = self.critic(next_obs_tensor)

                # Compute the advantage and store the rollout
                for agent in self.env.agents:
                    rollouts.append({
                        'state': obs[agent],
                        'action': actions[agent],
                        'log_prob': log_probs[agent],
                        'reward': rewards[agent],
                        'done': dones[agent],
                        'next_state': next_obs[agent],
                        'next_value': next_value,
                        # 'next_value': self.critic(next_obs[agent]),
                    })

                obs = next_obs
                done = any(dones.values())  # Assuming it's a multi-agent environment

            # Update the model after each episode
            self.update(rollouts)