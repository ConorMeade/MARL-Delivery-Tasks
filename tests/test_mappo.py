import copy

import numpy as np
import torch

from actor_critic import Actor, Critic
from mappo import MAPPO
from PickUpDropOffSimpleSpread import PickUpDropOffSimpleSpread

OBS_DIM = 12
ACT_DIM = 5


def _make_mappo():
    env = PickUpDropOffSimpleSpread(seed=42, max_cycles=10, num_agents=2, num_tasks=2)
    actor = Actor(obs_dim=OBS_DIM, act_dim=ACT_DIM)
    critic = Critic(obs_dim=OBS_DIM)
    return MAPPO(env, actor, critic)


def _make_rollouts(n=4):
    rollouts = []
    for _ in range(n):
        rollouts.append({
            'state': np.random.randn(OBS_DIM).astype(np.float32),
            'action': np.random.randint(0, ACT_DIM),
            'log_prob': torch.tensor(np.random.uniform(-2, 0), dtype=torch.float32),
            'reward': float(np.random.uniform(-1, 1)),
            'termination': False,
            'next_state': np.random.randn(OBS_DIM).astype(np.float32),
            'next_value': 0.0,
        })
    return rollouts


def test_compute_advantages_shapes_and_terminal_handling():
    mappo_agent = _make_mappo()

    rewards = torch.tensor([1.0, 1.0, 1.0])
    values = torch.tensor([0.5, 0.5, 0.5])
    next_values = torch.tensor([0.5, 0.5, 0.5])
    dones = torch.tensor([0.0, 0.0, 1.0])

    advantages, returns = mappo_agent.compute_advantages(rewards, values, next_values, dones)

    assert advantages.shape == rewards.shape
    assert returns.shape == rewards.shape
    # at the terminal step, the advantage should not bootstrap from next_value
    expected_last_advantage = rewards[-1] + mappo_agent.gamma * next_values[-1] * 0 - values[-1]
    assert torch.isclose(advantages[-1], expected_last_advantage)


def test_update_mappo_runs_and_updates_parameters():
    mappo_agent = _make_mappo()
    rollouts = _make_rollouts(n=4)

    params_before = [p.clone() for p in mappo_agent.actor.parameters()] + \
        [p.clone() for p in mappo_agent.critic.parameters()]

    mappo_agent.update_mappo(rollouts, next_obs={})

    params_after = list(mappo_agent.actor.parameters()) + list(mappo_agent.critic.parameters())

    changed = any(
        not torch.equal(before, after)
        for before, after in zip(params_before, params_after)
    )
    assert changed
