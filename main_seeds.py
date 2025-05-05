import torch
import numpy as np
from PickUpDropOffSimpleSpread import PickUpDropOffSimpleSpread
from mappo import MAPPO
from actor_critic import Actor, Critic
import matplotlib.pyplot as plt



def flatten_obs(obs_dict):
    parts = []
    for v in obs_dict.values():
        parts.append(np.asarray(v).flatten())
    return np.concatenate(parts)

def plot_rewards(cumulative_rewards, num_episodes, num_seeds, num_agents):
    all_ep_means = []
    for seed, episode_rewards in cumulative_rewards.items():
        # Convert each dict_values object to a list of floats and compute per-episode mean
        means = [np.mean(list(ep)) for ep in episode_rewards]
        all_ep_means.append(means)

    # Pad shorter lists with NaN if needed (in case seeds have different episode counts)
    max_len = max(len(m) for m in all_ep_means)
    for i in range(len(all_ep_means)):
        if len(all_ep_means[i]) < max_len:
            all_ep_means[i] = np.pad(all_ep_means[i], (0, max_len - len(all_ep_means[i])), constant_values=np.nan)

    all_rewards = np.array(all_ep_means)  # shape: (seeds, episodes)
    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        x=range(1, len(mean_rewards) + 1),
        y=mean_rewards,
        yerr=std_rewards,
        fmt='-o',
        capsize=3,
        label='Mean ± Std Dev'
    )
    plt.xlabel('Episode')
    plt.ylabel('Mean Cumulative Reward (per episode)')
    plt.title('Mean and Std Dev Across Seeds - 3 Agents, 2 Tasks')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("learning_curve_with_std_3_2.png")

def main():

    # per_agent_rewards_all = []  # Store per-agent rewards per episode
    # seeds = [42, 162, 120, 14, 45]
    seeds = [163, 11, 22]
    num_episodes =  50
    batch_size = 32
    cumulative_rewards = []
    cumulative_rewards = {}
    per_agent_rewards_all = []
    for s in seeds:
        cumulative_rewards[s] = []
        # seeds = [42, 162, 120, 14, 45]

        # Initialize the environment (GoalBasedSimpleSpread)
        base_env = PickUpDropOffSimpleSpread(seed=s, max_cycles=30, num_agents=3, num_tasks=2)  # Pass the number of tasks here (1 pickup/dropoff pair per agent)
        agent = base_env.agents[0]  # Just pick one agent
        obs_dim = base_env.observation_spaces(agent).shape[0]
        act_dim = base_env.action_spaces(agent).n

        # print(obs_dim)
        # print(act_dim)
        actor = Actor(obs_dim=obs_dim, act_dim=act_dim)
        critic = Critic(obs_dim=obs_dim)
        mappo_agent = MAPPO(base_env, actor, critic)

        # Training parameters
        # num_episodes = 30
        # batch_size = 32
        
        # Training loop
        # Reset env, generate rollouts
        # After 32 rollouts have been generated, call update() to improve policy
        for episode in range(num_episodes):
            obs = base_env.reset()  # Reset the environment and get initial observations
            episode_rewards = {agent: 0 for agent in base_env.agents}  # Initialize rewards
            rollouts = []  # Store rollouts for updating the model
            
            print(f'Starting Positions')
            for i in range(len(base_env.starting_positions)):
                print(f'Agent {i} X: {base_env.starting_positions[i][0]} Y: {base_env.starting_positions[i][1]}')
            print(f'Pickup locations: {base_env.pickups}')
            
            done_flags = {agent: False for agent in base_env.agents}

            # Start episode loop
            while not all(done_flags.values()):
                actions = {}
                log_probs = {}

                # Determine actions and log_probs for each agent in this step sequence
                # Call actor class to step based on actions selected with log_prob
                for agent in base_env.agents:
                    if done_flags[agent] or agent not in obs:
                        continue
                    obs_tensor = torch.tensor(obs[agent], dtype=torch.float32)
                    action, log_prob = mappo_agent.actor.act(obs_tensor)
                    actions[agent] = int(action)
                    log_probs[agent] = log_prob

                # Step the environment with the chosen actions
                next_obs, rewards, terminations, truncs, infos = base_env.step_pickup_drop(actions)

                # Store experiences in the rollouts buffer
                for agent in base_env.agents:
                    term = terminations.get(agent, False) or truncs.get(agent, False)
                    reward = rewards.get(agent, 0.0)
                    if terminations[agent]:
                        continue

                    # if not terminations[agent]:
                    if agent in next_obs:
                        state = obs[agent]
                        next_state = next_obs[agent]
                        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                        next_value = mappo_agent.critic(next_state_tensor).item()
                    else:
                        state = np.zeros(obs_dim)
                        next_state = np.zeros(obs_dim, dtype=np.float32) # zeros with same shape
                        next_value = torch.tensor(0.0)

                    action_roll = actions.get(agent, np.zeros(base_env.action_spaces(agent).shape, dtype=np.float32))
                    log_prob_roll = log_probs.get(agent, torch.tensor(0.0))
        
                    rollouts.append({
                        'state': state,
                        'action': actions[agent],
                        # 'action': action_roll,
                        'log_prob': log_prob_roll,
                        'reward': reward,
                        'termination': term,
                        'next_state': next_state,
                        'next_value': next_value,
                    })
                    # episode_rewards[agent] += rewards[agent]  

                obs = next_obs
                done_flags.update(terminations)

                # Update the model at specified intervals (full batch size)
                if len(rollouts) >= batch_size:
                    mappo_agent.update_mappo(rollouts, next_obs) 
                    rollouts = []
            
            for agent in base_env.agents:
                episode_rewards[agent] += rewards[agent]

            print(f"Episode {episode + 1}/{num_episodes}: Rewards = {episode_rewards}")
            # print(terminations)
            cumulative_rewards[s].append(episode_rewards.values())
            # cumulative_rewards.append((episode_rewards.values()))
            # per_agent_rewards_all.append(episode_rewards.copy())


    plot_rewards(cumulative_rewards, num_episodes, len(seeds), len(base_env.agents))

main()
