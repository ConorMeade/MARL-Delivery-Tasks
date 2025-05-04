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

def plot_rewards(cumulative_rewards, num_episodes, num_seeds):
    # all_rewards = np.array([
    #     [episode_rewards[ep] for ep in sorted(episode_rewards)]
    #     for episode_rewards in cumulative_rewards
    # ])


    all_rewards = np.array([
        list(episode) for episode in cumulative_rewards
    ])

    print(type(cumulative_rewards))
    print(len(cumulative_rewards))  # should be num_seeds
    print(type(cumulative_rewards[0]))
    print(len(cumulative_rewards[0]))  # should be num_episodes

    # rewards_across_seeds = [num_seeds, num_episodes]
    mean_rewards = all_rewards.mean(axis=0)
    std_rewards = all_rewards.std(axis=0)
    # std_rewards = np.std(all_rewards)


    plt.figure(figsize=(6, 4))
    plt.errorbar(range(len(mean_rewards)), mean_rewards, yerr=std_rewards, label='Mean ± Std Dev', fmt='-o', capsize=3)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Learning Curve with Standard Deviation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("learning_curve_with_std.png")
    # plt.plot(mean_rewards, label='Mean Reward')
    # plt.fill_between(np.arange(num_episodes), mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2, label='Standard Deviation')
    # plt.xlabel('Episode')
    # plt.ylabel('Cumulative Reward')
    # plt.title('Learning Curve Across Seeds (3 Seeds, 3 Actors)')
    # plt.legend()
    # # plt.savefig('Mean and Std Dev Rewards Diff Seeds.png') 
    # plt.close()  

    # Plotting mean and std deviation
    # plt.figure(figsize=(8, 6))
    # plt.plot(mean_rewards, label='Mean Reward')
    # plt.fill_between(range(num_episodes),
    #                 mean_rewards - std_rewards,
    #                 mean_rewards + std_rewards,
    #                 alpha=0.3, label='±1 Std Dev')
    # plt.xlabel('Episode')
    # plt.ylabel('Cumulative Reward')
    # plt.title('MAPPO Training Performance Across 5 Seeds')
    # plt.legend()
    # plt.grid(True)
    plt.savefig('mappo_performance_across_seeds.png')
    plt.close()

def main():

    # per_agent_rewards_all = []  # Store per-agent rewards per episode
    # seeds = [42, 162, 120, 14, 45]
    seeds = [163]
    num_episodes = 5
    batch_size = 32
    cumulative_rewards = []
    per_agent_rewards_all = []
    for s in seeds:

        # seeds = [42, 162, 120, 14, 45]

        # Initialize the environment (GoalBasedSimpleSpread)
        base_env = PickUpDropOffSimpleSpread(seed=s, max_cycles=20, num_tasks=2)  # Pass the number of tasks here (1 pickup/dropoff pair per agent)
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
            done = False
            episode_rewards = {agent: 0 for agent in base_env.agents}  # Initialize rewards
            rollouts = []  # Store rollouts for updating the model
            
            print(f'Starting Positions')
            for i in range(len(base_env.fixed_positions)):
                print(f'Agent {i} X: {base_env.fixed_positions[i][0]} Y: {base_env.fixed_positions[i][1]}')
            print(f'Pickup locations: {base_env.pickups}')
            print(f'Drop off Locations: {base_env.dropoffs}')
            
            if episode % 25 == 0:
                print(f'Episodes #{episode}')
            
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
                    if terminations[agent]:
                        continue

                    # if not terminations[agent]:
                    if agent in next_obs:
                        state = obs[agent]
                        next_state = next_obs[agent]
                        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                        next_value = mappo_agent.critic(next_state_tensor).squeeze()
                    else:
                        state = np.zeros(obs_dim)
                        next_state = np.zeros(obs_dim, dtype=np.float32) # zeros with same shape
                        next_value = torch.tensor(0.0)

                    action_roll = actions.get(agent, np.zeros(base_env.action_spaces(agent).shape, dtype=np.float32))
                    log_prob_roll = log_probs.get(agent, torch.tensor(0.0))
        
                    rollouts.append({
                        'state': state,
                        'action': action_roll,
                        'log_prob': log_prob_roll,
                        'reward': rewards[agent],
                        'termination': terminations[agent],
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
            cumulative_rewards.append((episode_rewards.values()))
            per_agent_rewards_all.append(episode_rewards.copy())


    plot_rewards(cumulative_rewards, num_episodes, len(seeds))

main()
