import numpy as np
import random
from collections import defaultdict
from pettingzoo.mpe import simple_spread_v3


class PickUpDropOffSimpleSpread:
    def __init__(self, seed, max_cycles, num_agents, num_tasks=5):
        self.num_agents = num_agents
        self.env = simple_spread_v3.parallel_env(
            render_mode="human",
            N=num_agents,
            local_ratio=0.5,
            max_cycles=max_cycles
        )
        self.seed = seed
        # self.env(seed)
        self.env.reset(seed=seed)
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

        self.starting_positions = None

        self.agent_goals = defaultdict(dict)
        for idx, agent in enumerate(self.agents):
            task_idx = random.randint(0, self.num_tasks - 1)
            self.agent_goals[agent] = {
                # 'pickup': self.pickups[task_idx],
                'pickup': self.pickups,
                'dropoff': self.dropoffs[task_idx],
                'reached_pickup': False,
                'reached_dropoff': False,
                'pickup_reward': False,
                'dropoff_reward': False,
                'goals_completed': 0,
                'pickups_visited': []
            }
        # Observation and action spaces for each agent
        self.observation_spaces = self.env.observation_space
        self.action_spaces = self.env.action_space
    
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


        # on reset, set starting location to same location each episode
        if self.starting_positions is None:
            self.starting_positions = []
            for _ in self.env.unwrapped.world.agents:
                pos = np.random.uniform(low=-0.8, high=0.8, size=(2,))
                self.starting_positions.append(pos)

        for i, agent in enumerate(self.env.unwrapped.world.agents):
            agent.state.p_po = self.starting_positions[i]
            agent.state.p_vel = np.zeros(2)


        self.agent_goals = defaultdict(dict)
        for idx, agent in enumerate(self.agents):
            task_idx = random.randint(0, self.num_tasks - 1)
            self.agent_goals[agent] = {
                # 'pickup': self.pickups[task_idx],
                'pickup': self.pickups,
                'dropoff': self.dropoffs[task_idx],
                'reached_pickup': False,
                'reached_dropoff': False,
                'pickup_reward': False,
                'dropoff_reward': False,
                'goals_completed': 0,
                'pickups_visited': []
            }
        self.step_count = 0
        obs_flat = {}

        for agent, raw_obs in obs[0].items():
            obs_flat[agent] = self._flatten_if_needed(raw_obs)

        return obs_flat

    def step_pickup_drop(self, actions):
        self.step_count += 1

        # Ensure some action is made, default - stay in current location
        for agent in self.agents:
            if agent not in actions:
                actions[agent] = 0

        for agent, action in actions.items():
            assert self.action_spaces(agent).contains(action), f"Invalid action {action} for {agent}"

        # Step the environment with the given actions
        observation, rewards, termination, truncs, infos = self.env.step(actions)

        obs_flat = {}
        for agent, raw_obs in observation.items():
            obs_flat[agent] = self._flatten_if_needed(raw_obs)
        
        # add rewards from pettingzoo step func 
        # clip rewards so they don't impact learning too much
        for agent, reward in rewards.items():
            if not self.agent_termination_flags[agent]:
                self.agent_rewards_out[agent] += reward
                # self.agent_rewards_out[agent] += np.clip(reward, -0.5, 0.5)
                # self.agent_rewards_out[agent] += np.clip(reward, -0.01, 0.01)

        # Reward calculation and goal tracking
        for agent in self.agents:
            if agent not in observation:
                continue  # skip agents no longer in the environment, agents that have term/trunc
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
            # only take first two elems for position
            pos = observation[agent][:2]
            goal = self.agent_goals[agent]

            if not goal['reached_pickup'] and not goal['pickup_reward']:
                # iterate through all pickup locations, 
                # if position is close enough, 
                # consider that agent to have reached the pickup location & completed a task
                for g_idx in range(len(goal['pickup'])):
                    # goal_index = goal['pickup'].index(g)
                    cur_goal_state = goal['pickup'][g_idx]
                    if np.linalg.norm(pos - cur_goal_state) < 0.15 and not self.is_visited(cur_goal_state, self.agent_goals[agent]['pickups_visited']):
                        print(f'{agent} Reached Pickup Location #{g_idx}')
                        goal['reached_pickup'] = True
                        goal['pickup_reward'] = True
                        # do not want to revisit pickup locations
                        # goal['pickups_visited'].append(cur_goal_state)
                        self.agent_infos_out[agent]['color'] = 'orange'
                        
                        self.agent_goals[agent]['pickups_visited'].append(cur_goal_state)

                        self.agent_goals[agent]['goals_completed'] += 1
                        if self.agent_goals[agent]['goals_completed'] < self.num_tasks:
                            self.agent_rewards_out[agent] += 100
                            goal['reached_pickup'] = False
                            goal['pickup_reward'] = False
                            self.agent_termination_flags[agent] = False

                        if self.agent_goals[agent]['goals_completed'] == self.num_tasks:
                            self.agent_rewards_out[agent] += 100
                            goal['reached_pickup'] = True
                            goal['pickup_reward'] = True
                            self.agent_termination_flags[agent] = True

        if self.step_count >= self.max_cycles:
            terminations = {agent: True for agent in self.agents}
            return obs_flat, dict(self.agent_rewards_out), terminations, dict(self.agent_truncs_out), dict(self.agent_infos_out)
        else:
            return obs_flat, dict(self.agent_rewards_out), dict(self.agent_termination_flags), dict(self.agent_truncs_out), dict(self.agent_infos_out)
    
    def is_visited(self, loc, visited_locations):
        return any(np.array_equal(loc, visited) for visited in visited_locations)
    
    def _flatten_if_needed(self, obs):
        """If obs is a dict, flatten it into a 1D array."""
        if isinstance(obs, dict):
            return np.concatenate([np.array(v).flatten() for v in obs.values()])
        else:
            return np.array(obs)