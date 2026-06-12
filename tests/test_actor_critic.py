import torch

from actor_critic import Actor, Critic

OBS_DIM = 12
ACT_DIM = 5


def test_actor_forward_output_shape():
    actor = Actor(obs_dim=OBS_DIM, act_dim=ACT_DIM)
    obs = torch.randn(OBS_DIM)
    logits = actor(obs)
    assert logits.shape == (ACT_DIM,)


def test_actor_act_returns_valid_action_and_log_prob():
    actor = Actor(obs_dim=OBS_DIM, act_dim=ACT_DIM)
    obs = torch.randn(OBS_DIM)
    action, log_prob = actor.act(obs)
    assert 0 <= int(action) < ACT_DIM
    assert torch.is_tensor(log_prob)
    assert log_prob.item() <= 0  # log of a probability is <= 0


def test_actor_evaluate_actions_matches_batch_size():
    actor = Actor(obs_dim=OBS_DIM, act_dim=ACT_DIM)
    batch = torch.randn(4, OBS_DIM)
    actions = torch.randint(0, ACT_DIM, (4,))
    log_probs = actor.evaluate_actions(batch, actions)
    assert log_probs.shape == (4,)


def test_actor_entropy_is_non_negative():
    actor = Actor(obs_dim=OBS_DIM, act_dim=ACT_DIM)
    batch = torch.randn(4, OBS_DIM)
    entropy = actor.get_entropy(batch)
    assert entropy.shape == (4,)
    assert torch.all(entropy >= 0)


def test_critic_forward_adds_batch_dim_for_1d_input():
    critic = Critic(obs_dim=OBS_DIM)
    obs = torch.randn(OBS_DIM)
    value = critic(obs)
    assert value.shape == (1,)


def test_critic_forward_handles_batched_input():
    critic = Critic(obs_dim=OBS_DIM)
    batch = torch.randn(4, OBS_DIM)
    values = critic(batch)
    assert values.shape == (4,)


def test_critic_accepts_numpy_input():
    import numpy as np

    critic = Critic(obs_dim=OBS_DIM)
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    value = critic(obs)
    assert value.shape == (1,)
