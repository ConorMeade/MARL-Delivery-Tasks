import numpy as np
import random

class PickUpDropOffSimpleSpread:
    def __init__(self, base_env, num_tasks=1):
        self.env = base_env
        self.env.reset()

        self.agents = self.env.agents
        self.num_tasks = num_tasks  # 1 pickup tasks and 1 drop off task

        self.pickups = None
        self.dropoffs = None
        self.agent_goals = {}
        self._setup_task_goals()

        self.observation_spaces = self.env.observation_spaces
        self.action_spaces = self.env.action_spaces

    def _setup_task_goals(self):
        # randomly select pickup and drop off task locations
        self.pickups = [np.random.uniform(-1, 1, size=(2,)) for _ in range(self.num_tasks)]
        self.dropoffs = [np.random.uniform(-1, 1, size=(2,)) for _ in range(self.num_tasks)]
        self.agent_goals = {}
        for agent in self.agents:
            task_idx = random.randint(0, self.num_tasks - 1)
            self.agent_goals[agent] = {
                'pickup': self.pickups[task_idx],
                'dropoff': self.dropoffs[task_idx],
                'reached_pickup': False,
                'reached_dropoff': False
            }

    def reset(self):
        obs = self.env.reset()
        self._setup_task_goals()
        return obs

    def step(self, actions):
        obs, rewards, dones, truncs, infos = self.env.step(actions)
        infos = infos or {agent: {} for agent in self.agents}

        for agent in self.agents:
            pos = self.env.state[agent]["p_pos"]
            goal = self.agent_goals[agent]

            if not goal['reached_pickup']:
                if np.linalg.norm(pos - goal['pickup']) < 0.1:
                    goal['reached_pickup'] = True
                    rewards[agent] += 1.0  # Reward for reaching pickup
                    infos[agent]['color'] = 'orange'
                else:
                    infos[agent]['color'] = 'red'
            elif not goal['reached_dropoff']:
                if np.linalg.norm(pos - goal['dropoff']) < 0.1:
                    goal['reached_dropoff'] = True
                    rewards[agent] += 2.0  # Reward for reaching dropoff
                    infos[agent]['color'] = 'green'
                else:
                    infos[agent]['color'] = 'orange'
            else:
                infos[agent]['color'] = 'green'

        return obs, rewards, dones, truncs, infos
