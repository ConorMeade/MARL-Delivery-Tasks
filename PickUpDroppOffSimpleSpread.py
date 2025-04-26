import numpy as np
import random
from collections import defaultdict
from pettingzoo.mpe import simple_spread_v3


class PickUpDropOffSimpleSpread:
    def __init__(self, seed, num_tasks=1):
        self.env = simple_spread_v3.env(render_mode="human")  # PettingZoo simple_spread_v3 environment
        self.env.reset(seed=seed)
        self.num_tasks = num_tasks
        # self.observations, self.infos = self.env
        self.agents = list(self.env.agents)
        print(self.agents)

        self.pickups = None
        self.dropoffs = None
        self.agent_goals = {}
        self._setup_task_goals()

        # Observation and action spaces for each agent
        self.observation_spaces = self.env.observation_space
        self.action_spaces = self.env.action_space

    def _setup_task_goals(self):
        # Setup random pickup and dropoff locations for each agent
        self.pickups = [np.random.uniform(-1, 1, size=(2,)) for _ in range(self.num_tasks)]
        self.dropoffs = [np.random.uniform(-1, 1, size=(2,)) for _ in range(self.num_tasks)]

        # print(self.pickups)
        # print(self.dropoffs)
        
        self.agent_goals = defaultdict(dict)
        for idx, agent in enumerate(self.agents):
            task_idx = random.randint(0, self.num_tasks - 1)
            self.agent_goals[agent] = {
                'pickup': self.pickups[task_idx],
                'dropoff': self.dropoffs[task_idx],
                'reached_pickup': False,
                'reached_dropoff': False
            }

    def reset(self):
        self.env.reset()  # Reset PettingZoo environment
        self._setup_task_goals()  # Setup new task goals after reset
        obs = {agent: self.env.observe(agent) for agent in self.agents}
        return obs

    def step(self, actions):
        # Step the environment with the given actions
        next_obs, rewards, dones, truncs, infos = self.env.step(actions)

        # Reward calculation and goal tracking
        for agent in self.agents:
            pos = self.env.state[agent]["p_pos"]
            goal = self.agent_goals[agent]

            if not goal['reached_pickup']:
                if np.linalg.norm(pos - goal['pickup']) < 0.1:
                    goal['reached_pickup'] = True
                    rewards[agent] += 1.0  # Reward for reaching pickup
                    infos[agent]['color'] = 'orange'  # Visual indication of pickup
                else:
                    infos[agent]['color'] = 'red'  # Still at pickup stage
            elif not goal['reached_dropoff']:
                if np.linalg.norm(pos - goal['dropoff']) < 0.1:
                    goal['reached_dropoff'] = True
                    rewards[agent] += 2.0  # Reward for reaching dropoff
                    infos[agent]['color'] = 'green'  # Visual indication of dropoff
                else:
                    infos[agent]['color'] = 'orange'  # Pickup reached, now heading to dropoff
            else:
                infos[agent]['color'] = 'green'  # Goal achieved (both pickup and dropoff)

        return next_obs, rewards, dones, truncs, infos