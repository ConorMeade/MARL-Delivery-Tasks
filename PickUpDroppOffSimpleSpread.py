import numpy as np
from pettingzoo.mpe import simple_spread_v3
from gymnasium import spaces

class GoalBasedSimpleSpread:
    def __init__(self, num_agents=3, max_cycles=25):
        self.env = simple_spread_v3.parallel_env(N=num_agents, max_cycles=max_cycles, render_mode="human")
        self.env.reset()
        self.num_agents = num_agents
        self.agents = self.env.agents
        self.max_cycles = max_cycles

        self.pickup_locations = {agent: np.random.uniform(-1, 1, size=(2,)) for agent in self.agents}
        self.dropoff_locations = {agent: np.random.uniform(-1, 1, size=(2,)) for agent in self.agents}
        self.pickup_complete = {agent: False for agent in self.agents}

        obs_space = self.env.observation_space(self.agents[0])
        self.observation_spaces = {agent: spaces.Box(low=-np.inf, high=np.inf, shape=(obs_space.shape[0] + 2,), dtype=np.float32) for agent in self.agents}
        self.action_spaces = {agent: self.env.action_space(agent) for agent in self.agents}

    def reset(self):
        obs = self.env.reset()
        self.pickup_locations = {agent: np.random.uniform(-1, 1, size=(2,)) for agent in self.agents}
        self.dropoff_locations = {agent: np.random.uniform(-1, 1, size=(2,)) for agent in self.agents}
        self.pickup_complete = {agent: False for agent in self.agents}
        return self._augment_obs(obs)

    def step(self, actions):
        obs, _, terminated, truncated, info = self.env.step(actions)

        rewards = {}
        for agent in self.agents:
            agent_pos = self.env.env.state[agent].p_pos
            goal = self.dropoff_locations[agent] if self.pickup_complete[agent] else self.pickup_locations[agent]
            dist = np.linalg.norm(agent_pos - goal)

            rewards[agent] = -dist

            if not self.pickup_complete[agent] and dist < 0.1:
                self.pickup_complete[agent] = True

        return self._augment_obs(obs), rewards, terminated, truncated, info

    def _augment_obs(self, obs_dict):
        new_obs = {}
        for agent in self.agents:
            goal = self.dropoff_locations[agent] if self.pickup_complete[agent] else self.pickup_locations[agent]
            new_obs[agent] = np.concatenate([obs_dict[agent], goal])
        return new_obs
