import numpy as np
import pytest

from PickUpDropOffSimpleSpread import PickUpDropOffSimpleSpread


@pytest.fixture
def env():
    return PickUpDropOffSimpleSpread(seed=42, max_cycles=10, num_agents=2, num_tasks=2)


def test_init_creates_one_agent_goal_per_agent(env):
    assert len(env.agents) == 2
    assert set(env.agent_goals.keys()) == set(env.agents)


def test_single_shared_pickup_location(env):
    # Only one pickup location should exist, shared across agents/tasks,
    # and fixed so the task is consistent across episodes/seeds
    assert len(env.pickups) == 1
    assert np.array_equal(env.pickups[0], np.array([0.5, 0.5]))
    for agent in env.agents:
        assert len(env.agent_goals[agent]['pickup']) == 1
        assert np.array_equal(env.agent_goals[agent]['pickup'][0], env.pickups[0])


def test_reset_returns_flat_observations_for_each_agent(env):
    obs = env.reset()
    assert set(obs.keys()) == set(env.agents)
    for agent in env.agents:
        assert obs[agent].ndim == 1
        assert obs[agent].shape[0] == env.observation_spaces(agent).shape[0]


def test_reset_initializes_fresh_goal_state(env):
    env.reset()
    for agent in env.agents:
        goal = env.agent_goals[agent]
        assert goal['goals_completed'] == 0
        assert goal['reached_pickup'] is False
        assert goal['pickup_reward'] is False


def test_step_pickup_drop_returns_expected_structures(env):
    env.reset()
    actions = {agent: 0 for agent in env.agents}  # 0 = no-op
    obs, rewards, terminations, truncs, infos = env.step_pickup_drop(actions)

    assert set(obs.keys()) == set(env.agents)
    assert set(rewards.keys()) == set(env.agents)
    for agent in env.agents:
        assert isinstance(rewards[agent], float)


def test_reaching_pickup_location_grants_reward_and_progress(env):
    env.reset()
    agent = env.agents[0]
    pickup_loc = env.pickups[0]

    # Teleport the agent directly onto the pickup location (position is at
    # observation indices 2:4; velocity 0 so the step doesn't move it away)
    env.env.unwrapped.world.agents[0].state.p_pos = np.array(pickup_loc, dtype=np.float32)
    env.env.unwrapped.world.agents[0].state.p_vel = np.zeros(2, dtype=np.float32)

    actions = {a: 0 for a in env.agents}
    obs, rewards, terminations, truncs, infos = env.step_pickup_drop(actions)

    # +30 pickup bonus dominates the small per-step penalty from the base env
    assert rewards[agent] > 25
    assert env.agent_goals[agent]['goals_completed'] == 1


def test_episode_terminates_at_max_cycles(env):
    env.reset()
    actions = {a: 0 for a in env.agents}

    terminations = {a: False for a in env.agents}
    steps = 0
    while not all(terminations.values()) and steps < env.max_cycles:
        _, _, terminations, _, _ = env.step_pickup_drop(actions)
        steps += 1

    assert steps == env.max_cycles
    assert all(terminations.values())


def test_invalid_action_raises_assertion(env):
    env.reset()
    actions = {a: 99 for a in env.agents}  # outside Discrete(5) action space
    with pytest.raises(AssertionError):
        env.step_pickup_drop(actions)
