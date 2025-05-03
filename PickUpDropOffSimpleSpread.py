import numpy as np
import random
from collections import defaultdict
from pettingzoo.mpe import simple_spread_v3 # type: ignore
from enum import Enum

# class PickUpStatus(Enum):
#     HASPICKUP = 1
#     DROPPEDOFF = 2
#     COMPLETEDTASKS = 3
#     NOPICKUP = 4

class PickUpDropOffSimpleSpread:
    def __init__(self, seed, max_cycles, num_tasks=2):
        self.env = simple_spread_v3.parallel_env(
            render_mode="human",
            N=3,
            local_ratio=0.5,
            max_cycles=max_cycles
        )
        self.seed = seed
        # self.env(seed)
        self.env.reset(seed=self.seed)
        self.num_tasks = num_tasks
        # self.observations, self.infos = self.env
        self.agents = list(self.env.agents)
        self.agent_termination_flags = {agent: False for agent in self.agents}
        self.agent_rewards_out = {agent: 0.0 for agent in self.agents}
        self.agent_truncs_out = {agent: False for agent in self.agents}
        self.agent_infos_out = {agent: {} for agent in self.agents}

        self.step_count = 0
        self.max_cycles = max_cycles

        self.pickups = None
        self.dropoffs = None
        self.agent_goals = {}
        self._setup_task_goals()

        self.fixed_positions = None

        self.agent_goals = defaultdict(dict)
        for idx, agent in enumerate(self.agents):
            task_idx = random.randint(0, self.num_tasks - 1)
            self.agent_goals[agent] = {
                'pickup': self.pickups[task_idx],
                'dropoff': self.dropoffs[task_idx],
                'reached_pickup': False,
                'reached_dropoff': False,
                'pickup_reward': False,
                'dropoff_reward': False,
                'goals_completed': 0
            }
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

    def reset(self):
        obs = self.env.reset(seed=self.seed)
        self.agents = list(self.env.agents)  # Update agents list after reset
        self.agent_termination_flags = {agent: False for agent in self.agents}
        self.agent_rewards_out = {agent: 0.0 for agent in self.agents}
        self.agent_truncs_out = {agent: False for agent in self.agents}
        self.agent_infos_out = {agent: {} for agent in self.agents}
        # self._setup_task_goals()

        if self.fixed_positions is None:
            self.fixed_positions = []
            for _ in self.env.unwrapped.world.agents:
                pos = np.random.uniform(low=-0.8, high=0.8, size=(2,))
                self.fixed_positions.append(pos)

        for i, agent in enumerate(self.env.unwrapped.world.agents):
            agent.state.p_po = self.fixed_positions[i]
            agent.state.p_vel = np.zeros(2)


        self.agent_goals = defaultdict(dict)
        for idx, agent in enumerate(self.agents):
            task_idx = random.randint(0, self.num_tasks - 1)
            self.agent_goals[agent] = {
                'pickup': self.pickups[task_idx],
                'dropoff': self.dropoffs[task_idx],
                'reached_pickup': False,
                'reached_dropoff': False,
                'pickup_reward': False,
                'dropoff_reward': False,
                'goals_completed': 0
            }
        self.step_count = 0
        obs_flat = {}

        for agent, raw_obs in obs[0].items():
            obs_flat[agent] = self._flatten_if_needed(raw_obs)

        return obs_flat

    def step_pickup_drop(self, actions):
        self.step_count += 1
        # Ensure all actions are int
        # actions = {agent: int(action) for agent, action in actions.items()}
        # actions = {agent: int(self.action_spaces(agent).sample()) for agent in self.agents}

        for agent in self.agents:
            if agent not in actions:
                actions[agent] = 0

        for agent, action in actions.items():
            assert self.action_spaces(agent).contains(action), f"Invalid action {action} for {agent}"

        # Step the environment with the given actions
        observation, rewards, termination, truncs, infos = self.env.step(actions)

        if rewards['agent_0'] != rewards['agent_1'] or rewards['agent_1'] != rewards['agent_2'] or rewards['agent_0'] != rewards['agent_2']:
            print('rewards changed') 

        # if rewards['agent_0'] < -1.0:
            # print('negativeeee')
        obs_flat = {}
        for agent, raw_obs in observation.items():
            obs_flat[agent] = self._flatten_if_needed(raw_obs)
        # next_obs, rewards, termination, truncs, infos = self.env.last()
        
        # add rewards from pettingzoo step func 
        # clip rewards so they don't impact learning too much
        for agent, reward in rewards.items():
            self.agent_rewards_out[agent] += np.clip(reward, -0.01, 0.01)

        # Reward calculation and goal tracking
        for agent in self.agents:
            if agent not in observation:
                continue  # skip agents no longer in the environment
            # REF
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

            # current status
            # if goal['reached_pickup'] and not goal['reached_dropoff']:
                # self.agent_infos_out[agent]['color'] = 'orange'
            if not goal['reached_pickup'] and not goal['pickup_reward']:
                if np.linalg.norm(pos - goal['pickup']) < 0.2:
                    print(f'{agent} reached goal 1')
                    goal['reached_pickup'] = True
                    goal['pickup_reward'] = True
                    self.agent_rewards_out[agent] += 1
                    self.agent_infos_out[agent]['color'] = 'orange'
                    # self.agent_infos_out['reached_pickup'] = True
                # else:
                #     self.agent_infos_out[agent]['color'] = 'red'
            elif not goal['reached_dropoff'] and not goal['dropoff_reward']:
                if np.linalg.norm(pos - goal['dropoff']) < 0.2:
                    print(f'{agent} reached goal 2')
                    goal['reached_dropoff'] = True
                    goal['dropoff_reward'] = True
                    self.agent_infos_out[agent]['color'] = 'green'
                    self.agent_goals[agent]['goals_completed'] += 1
                    self.agent_termination_flags[agent] = True
                    # Do a per task reset
                    # So, if more tasks need to be completed, make it so we can
                    # still enter these if statements again
                    if self.agent_goals[agent]['goals_completed'] < self.num_tasks:
                        self.agent_rewards_out[agent] += 2
                        goal['reached_pickup'] = False
                        goal['reached_dropoff'] = False
                        goal['pickup_reward'] = False
                        goal['dropoff_reward'] = False
                        self.agent_termination_flags[agent] = False
                    if self.agent_goals[agent]['goals_completed'] == self.num_tasks:
                        self.agent_rewards_out[agent] += 2
                    # if self.agent_goals[agent]['goals_completed'] == self.num_tasks:

                # else:
                #     self.agent_infos_out[agent]['color'] = 'orange'
            # else:
            #     self.agent_infos_out[agent]['color'] = 'green'

        if self.step_count >= self.max_cycles:
            terminations = {agent: True for agent in self.agents}
            return obs_flat, dict(self.agent_rewards_out), terminations, dict(self.agent_truncs_out), dict(self.agent_infos_out)
        else:
            return obs_flat, dict(self.agent_rewards_out), dict(self.agent_termination_flags), dict(self.agent_truncs_out), dict(self.agent_infos_out)
    
    def _flatten_if_needed(self, obs):
        """If obs is a dict, flatten it into a 1D array."""
        if isinstance(obs, dict):
            return np.concatenate([np.array(v).flatten() for v in obs.values()])
        else:
            return np.array(obs)