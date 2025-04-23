import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from actor_critic import Actor, Critic

class MAPPO:
    def __init__(self, env, gamma=0.99, lr=1e-3, clip_eps=0.2):
        self.env = env
        self.gamma = gamma
        self.clip_eps = clip_eps

        obs_dim = env.observation_spaces[env.agents[0]].shape[0]
        act_dim = env.action_spaces[env.agents[0]].n

        self.actors = {agent: Actor(obs_dim, act_dim) for agent in env.agents}
        self.critics = {agent: Critic(obs_dim) for agent in env.agents}
        self.optimizers = {
            agent: optim.Adam(
                list(self.actors[agent].parameters()) + list(self.critics[agent].parameters()), lr=lr
            )
            for agent in env.agents
        }

    def collect_trajectory(self, horizon=100):
        memory = {agent: {'obs': [], 'actions': [], 'log_probs': [], 'rewards': [], 'values': []} for agent in self.env.agents}
        obs = self.env.reset()

        for _ in range(horizon):
            actions = {}
            for agent in self.env.agents:
                obs_tensor = torch.tensor(obs[agent], dtype=torch.float32)
                action, log_prob = self.actors[agent].sample_action(obs_tensor)
                value = self.critics[agent](obs_tensor).item()

                memory[agent]['obs'].append(obs_tensor)
                memory[agent]['actions'].append(action)
                memory[agent]['log_probs'].append(log_prob)
                memory[agent]['values'].append(value)

                actions[agent] = action

            next_obs, rewards, dones, truncs, infos = self.env.step(actions)

            for agent in self.env.agents:
                memory[agent]['rewards'].append(rewards[agent])

            obs = next_obs

        return memory

    def compute_returns(self, rewards, values, gamma):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32), torch.tensor(values, dtype=torch.float32)

    def update(self, memory):
        for agent in self.env.agents:
            obs = torch.stack(memory[agent]['obs'])
            actions = torch.tensor(memory[agent]['actions'])
            old_log_probs = torch.tensor(memory[agent]['log_probs'])
            returns, values = self.compute_returns(memory[agent]['rewards'], memory[agent]['values'], self.gamma)

            advantages = returns - values

            for _ in range(4):  # PPO epochs
                probs = self.actors[agent](obs)
                dist_probs = probs.gather(1, actions.unsqueeze(1)).squeeze(1)
                new_log_probs = torch.log(dist_probs + 1e-8)

                ratio = torch.exp(new_log_probs - old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

                value_preds = self.critics[agent](obs).squeeze(1)
                value_loss = (returns - value_preds).pow(2).mean()

                loss = policy_loss + 0.5 * value_loss

                self.optimizers[agent].zero_grad()
                loss.backward()
                self.optimizers[agent].step()
