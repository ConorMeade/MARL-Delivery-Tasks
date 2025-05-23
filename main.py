import torch
import numpy as np
from PickUpDropOffSimpleSpread import PickUpDropOffSimpleSpread
from mappo import MAPPO
from actor_critic import Actor, Critic
import matplotlib.pyplot as plt
from collections import defaultdict


def flatten_obs(obs_dict):
    parts = []
    for v in obs_dict.values():
        parts.append(np.asarray(v).flatten())
    return np.concatenate(parts)

def plot_rewards(cumulative_rewards, per_agent_rewards, num_agents, num_tasks):
    plt.figure(figsize=(8, 5))
    plt.plot(cumulative_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward (All Agents)')
    plt.title(f'Cumulative Ep Rewards - {num_agents} Agents, {num_tasks} Tasks')
    plt.savefig(f'learning_curves/cumulative_rewards_{num_agents}_agents_{num_tasks}_300episodes.png')
    plt.close()

    # agent_names = list(per_agent_rewards[0].keys())

    # for agent in agent_names:
    #     agent_rewards = [episode_rewards[agent] for episode_rewards in per_agent_rewards]
    #     plt.plot(agent_rewards, label=agent)

    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    # plt.title('Per-Agent Cumulative Rewards')
    # plt.legend()
    # plt.savefig(f'learning_curves/per_agent_rewards_seed42_{num_agents}_agents_{num_tasks}_tasks_all_locations.png') 
    # plt.close()  

def main():
    cumulative_rewards = []
    per_agent_rewards_all = []
    seeds = [42, 162, 120, 14, 45]

    # Initialize the environment (PickUpDropOffSimpleSpread)
    base_env = PickUpDropOffSimpleSpread(seed=42, max_cycles=20, num_agents=3, num_tasks=4)  # Pass the number of tasks here (1 pickup/dropoff pair per agent)
    agent = base_env.agents[0]  # Just pick one agent
    obs_dim = base_env.observation_spaces(agent).shape[0]
    act_dim = base_env.action_spaces(agent).n

    actor = Actor(obs_dim=obs_dim, act_dim=act_dim)
    critic = Critic(obs_dim=obs_dim)
    mappo_agent = MAPPO(base_env, actor, critic)

    # Training parameters
    num_episodes = 300
    batch_size = 16
    
    # Training loop
    # Reset env, generate rollouts
    # After 16 rollouts have been generated, call update() to improve policy
    for episode in range(num_episodes):
        obs = base_env.reset()  # Reset the environment and get initial observations
        episode_rewards = {agent: 0 for agent in base_env.agents}  # Initialize rewards
        rollouts = []  # Rollouts for mappo policy update
        
        print(f'Starting Positions')
        for i in range(len(base_env.starting_positions)):
            print(f'Agent {i} X: {base_env.starting_positions[i][0]} Y: {base_env.starting_positions[i][1]}')
        print(f'Pickup locations: {base_env.pickups}')

        done_flags = {agent: False for agent in base_env.agents}

        rewards_per_agent = defaultdict(float)
        # Start episode loop
        while not all(done_flags.values()):
            # if any(done_flags.values()):
                # print(done_flags)
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
                rewards_per_agent[agent] += reward
                if terminations[agent]:
                    continue

                # generate rollout values, evaluate state with critic, if no agent due to termination, return zeros
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
                    # 'action': action_roll,
                    'action': actions[agent],
                    'log_prob': log_prob_roll,
                    'reward': reward,
                    'termination': term,
                    'next_state': next_state,
                    'next_value': next_value,
                })

            for agent in base_env.agents:
                episode_rewards[agent] += rewards[agent]  
            

            obs = next_obs
            done_flags.update(terminations)

            # Update the model at specified intervals
            if len(rollouts) >= batch_size:
                # print('updating policy')
                # Update model using rollouts after we reach a full batch size
                mappo_agent.update_mappo(rollouts, next_obs) 
                rollouts = []

        # for agent in base_env.agents:
        #     episode_rewards[agent] += rewards[agent]  

        print(f"Episode {episode + 1}/{num_episodes}: Rewards = {episode_rewards}")
        # print(terminations)
        cumulative_rewards.append(sum(episode_rewards.values()))
        per_agent_rewards_all.append(episode_rewards.copy())


    plot_rewards(cumulative_rewards, per_agent_rewards_all, num_agents=base_env.num_agents, num_tasks=base_env.num_tasks)

main()
