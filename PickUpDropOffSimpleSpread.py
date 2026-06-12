import numpy as np
import random
import types
import pygame
from collections import defaultdict
from pettingzoo.mpe import simple_spread_v3


class PickUpDropOffSimpleSpread:
    def __init__(self, seed, max_cycles, num_agents, num_tasks=5):
        self.num_agents = num_agents
        self.env = simple_spread_v3.parallel_env(
            render_mode="human",
            N=num_agents,
            local_ratio=0.1,
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
                'goals_completed': 0
            }
        # Observation and action spaces for each agent
        self.observation_spaces = self.env.observation_space
        self.action_spaces = self.env.action_space

        self._patch_render_with_pickup()

    def _patch_render_with_pickup(self):
        """Overlay a marker for the pickup location on top of the
        default simple_spread rendering, without affecting reward
        computation (the pickup point is not added as a Landmark)."""
        base_env = self.env.unwrapped
        original_draw = base_env.draw
        pickup_loc = self.pickups[0]

        def draw_with_pickup(render_env):
            original_draw()

            cam_range = np.max(np.abs(np.array(
                [entity.state.p_pos for entity in render_env.world.entities]
            )))

            x, y = pickup_loc
            y *= -1  # match the y-flip used for entities in draw()
            x = (x / cam_range) * render_env.width // 2 * 0.9 + render_env.width // 2
            y = (y / cam_range) * render_env.height // 2 * 0.9 + render_env.height // 2

            pygame.draw.circle(render_env.screen, (255, 215, 0), (x, y), 12, 3)

        base_env.draw = types.MethodType(draw_with_pickup, base_env)

    def _setup_task_goals(self):
        # Single shared pickup location, fixed across all episodes/seeds so
        # every run trains on the same task
        self.pickups = [np.array([0.5, 0.5])]
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
                'goals_completed': 0
            }
        self.step_count = 0
        obs_flat = {}

        for agent, raw_obs in obs[0].items():
            obs_flat[agent] = self._flatten_if_needed(raw_obs)

        return obs_flat

    def step_pickup_drop(self, actions):
        self.step_count += 1
        # reset rewards at each step for an agent, only one single ep rewards for rollouts
        self.agent_rewards_out = {agent: 0.0 for agent in self.agents}

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
            #     x_vel, y_vel,                # ← 0–1 : velocity
            #     x_pos, y_pos,                # ← 2–3 : actual (global) position
            #     rel_landmark0_x, rel_landmark0_y,   # ← 4–5 : relative to landmark 0
            #     rel_landmark1_x, rel_landmark1_y,   # ← 6–7
            #     rel_landmark2_x, rel_landmark2_y,   # ← 8–9
            #     rel_agent1_x, rel_agent1_y,         # ← 10–11 : relative to other agent 1
            #     rel_agent2_x, rel_agent2_y,         # ← 12–13 : relative to other agent 2
            #     (maybe padding or comms)     # ← 14–17
            # ]
            # global position is at indices 2:4 (indices 0:2 are velocity)
            pos = observation[agent][2:4]
            goal = self.agent_goals[agent]

            if not goal['reached_pickup'] and not goal['pickup_reward']:
                # single shared pickup location;
                # if position is close enough, consider that agent to have
                # reached the pickup location & completed a task
                cur_goal_state = goal['pickup'][0]
                if np.linalg.norm(pos - cur_goal_state) < 0.05:
                    print(f'{agent} Reached Pickup Location')
                    goal['reached_pickup'] = True
                    goal['pickup_reward'] = True
                    self.agent_infos_out[agent]['color'] = 'orange'

                    self.agent_goals[agent]['goals_completed'] += 1
                    self.agent_rewards_out[agent] += 30

                    if self.agent_goals[agent]['goals_completed'] < self.num_tasks:
                        # allow revisiting the single pickup location for the next task
                        goal['reached_pickup'] = False
                        goal['pickup_reward'] = False
                        self.agent_termination_flags[agent] = False
                    else:
                        self.agent_termination_flags[agent] = True

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