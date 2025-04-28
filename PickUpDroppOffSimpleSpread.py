import numpy as np
import random
from collections import defaultdict
from pettingzoo.mpe import simple_spread_v3 # type: ignore


class PickUpDropOffSimpleSpread:
    def __init__(self, seed, num_tasks=1):
        self.env = simple_spread_v3.parallel_env(
            render_mode="human",
            N=3,
            local_ratio=0.5,
            max_cycles=200,
            continuous_actions=False
        )  
        self.env.reset(seed=seed)
        self.num_tasks = num_tasks
        # self.observations, self.infos = self.env
        self.agents = list(self.env.agents)
        # print(self.agents)

        self.pickups = None
        self.dropoffs = None
        self.agent_goals = {}
        self._setup_task_goals()

        # Observation and action spaces for each agent
        self.observation_spaces = self.env.observation_space
        self.action_spaces = self.env.action_space


    def render(self):
        """Forward the render call to the base environment."""
        self.env.render()

    
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

    # def reset(self):
    #     obs = self.env.reset()  # Reset PettingZoo environment
    #     self.agents = list(self.env.observation_spaces.keys())
    #     self._setup_task_goals()  # Setup new task goals after reset
    #     # obs = {agent: self.env.observe(agent) for agent in self.agents}
    #     return obs

    def reset(self):
        obs = self.env.reset()
        self.agents = list(self.env.agents)  # Update agents list after reset

        obs_flat = {}
        # print(obs)
        for agent, raw_obs in obs[0].items():
            obs_flat[agent] = self._flatten_if_needed(raw_obs)

        return obs_flat

    def step_pickup_drop(self, actions):

        # Ensure all actions are int
        # actions = {agent: int(action) for agent, action in actions.items()}
        actions = {agent: int(self.action_spaces(agent).sample()) for agent in self.agents}

    
        # print(actions)

        for agent, action in actions.items():
            assert self.action_spaces(agent).contains(action), f"Invalid action {action} for {agent}"

        # print(self.action_spaces('agent_0'))
        # Step the environment with the given actions
        observation, rewards, termination, truncs, infos = self.env.step(actions)

        obs_flat = {}
        for agent, raw_obs in observation.items():
            obs_flat[agent] = self._flatten_if_needed(raw_obs)
        # next_obs, rewards, termination, truncs, infos = self.env.last()

        rewards_out = {agent: 0.0 for agent in self.agents}
        term_out = {agent: False for agent in self.agents}
        truncs_out = {agent: False for agent in self.agents}
        infos_out = {agent: {} for agent in self.agents}


        # Reward calculation and goal tracking
        for agent in self.agents:
            if agent not in observation:
                continue  # skip agents no longer in the environment
            # observation[agent] = [
            #     x_pos, y_pos,                # ← 0–1 : actual (global) position
            #     x_vel, y_vel,                # ← 2–3 : velocity
            #     rel_landmark0_x, rel_landmark0_y,   # ← 4–5 : relative to landmark 0
            #     rel_landmark1_x, rel_landmark1_y,   # ← 6–7
            #     rel_landmark2_x, rel_landmark2_y,   # ← 8–9
            #     rel_agent1_x, rel_agent1_y,         # ← 10–11 : relative to other agent 1
            #     rel_agent2_x, rel_agent2_y,         # ← 12–13 : relative to other agent 2
            #     (maybe padding or comms)     # ← 14–17
            # ]
            pos = observation[agent][:2]
            # pos = self.env.state(agent)["p_pos"]
            goal = self.agent_goals[agent]

            if not goal['reached_pickup']:
                if np.linalg.norm(pos - goal['pickup']) < 0.7:
                    print(f'{agent} reached goal 1')
                    goal['reached_pickup'] = True
                    rewards_out[agent] += 1.0
                    infos_out[agent]['color'] = 'orange'
                else:
                    infos_out[agent]['color'] = 'red'
            elif not goal['reached_dropoff']:
                if np.linalg.norm(pos - goal['dropoff']) < 0.7:
                    print(f'{agent} reached goal 2')
                    goal['reached_dropoff'] = True
                    rewards_out[agent] += 2.0
                    infos_out[agent]['color'] = 'green'
                    term_out[agent] = True
                else:
                    infos_out[agent]['color'] = 'orange'
            else:
                infos_out[agent]['color'] = 'green'


            term_out[agent] = termination.get(agent, False)
            truncs_out[agent] = truncs.get(agent, False)  # Ensure truncs are passed

        # return all_agent_status
        return obs_flat, rewards_out, term_out, truncs_out, infos_out
        # return next_obs, rewards, termination, truncs, infos
    
    def _flatten_if_needed(self, obs):
        """If obs is a dict, flatten it into a 1D array."""
        if isinstance(obs, dict):
            return np.concatenate([np.array(v).flatten() for v in obs.values()])
        else:
            return np.array(obs)
        
    # def compute_advantage(self, rollouts, gamma=0.99, lam=0.95):
    #     advantages = []
    #     last_gae_lam = 0
    #     for r in reversed(rollouts):
    #         reward = r['reward']
    #         value = r['value']  # Placeholder for value prediction
    #         next_value = r['next_value']  # Placeholder for next value (last timestep should be 0)
    #         delta = reward + gamma * next_value - value
    #         last_gae_lam = delta + gamma * lam * last_gae_lam
    #         advantages.append(last_gae_lam)
    #     return advantages[::-1] 