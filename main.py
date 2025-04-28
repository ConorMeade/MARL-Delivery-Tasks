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

def plot_rewards(cumulative_rewards, per_agent_rewards):
    plt.plot(cumulative_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward (All Agents)')
    plt.title('Cumulative Episode Rewards')
    plt.savefig('cumulative_rewards.png')
    plt.close()

    agent_names = list(per_agent_rewards[0].keys())

    for agent in agent_names:
        agent_rewards = [episode_rewards[agent] for episode_rewards in per_agent_rewards]
        plt.plot(agent_rewards, label=agent)

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Per-Agent Cumulative Rewards')
    plt.legend()
    plt.savefig('per_agent_rewards.png') 
    plt.close()  

def main():
    cumulative_rewards = []  # Total reward across all agents, per episode
    per_agent_rewards_all = []  # Store per-agent rewards per episode

    # Initialize the environment (GoalBasedSimpleSpread)
    base_env = PickUpDropOffSimpleSpread(seed=42, max_cycles=100,num_tasks=1)  # Pass the number of tasks here (1 pickup/dropoff pair per agent)
    # sample_obs = base_env.reset()
    # sample_flat_obs = flatten_obs(sample_obs[0])
    agent = base_env.agents[0]  # Just pick one agent
    obs_dim = base_env.observation_spaces(agent).shape[0]
    act_dim = base_env.action_spaces(agent).n


    # obs_dim = sample_flat_obs.shape[0]  # ← dynamic based on env
    # act_dim = base_env.action_spaces[base_env.agents[0]].n
    # print(base_env.observation_spaces(0))
    # Define observation and action dimensions
    # obs_dim = base_env.observation_spaces(base_env.agents[0]).shape[0]  # Size of observation space for an agent
    # act_dim = base_env.action_spaces(base_env.agents[0]).n  # Size of action space for an agent

    print(obs_dim)
    print(act_dim)
    # Instantiate Actor and Critic separately
    actor = Actor(obs_dim=obs_dim, act_dim=act_dim)  # Instantiate the Actor
    critic = Critic(obs_dim=obs_dim)  # Instantiate the Critic

    # Initialize MAPPO model
    mappo_agent = MAPPO(base_env, actor, critic)

    # Training parameters
    num_episodes = 250
    batch_size = 32
    update_interval = 10  # Update the model after this many steps

    for episode in range(num_episodes):
        obs = base_env.reset()  # Reset the environment and get initial observations
        done = False
        episode_rewards = {agent: 0 for agent in base_env.agents}  # Initialize rewards
        rollouts = []  # Store rollouts for updating the model

        if episode % 25 == 0:
            print(f'Episodes #{episode}')
        
        done_flags = {agent: False for agent in base_env.agents}

        # Start episode loop
        while not all(done_flags.values()):
            # if any(done_flags.values()):
                # print(done_flags)
            actions = {}
            log_probs = {}

            episode_reward_single = 0
            # Collect actions and log_probs for each agent
            # for agent in range(len(base_env.agents)):
            for agent in base_env.agents:
                # print(agent)
                if done_flags[agent] or agent not in obs:
                    continue
                # obs_tensor = torch.tensor(obs[agent], dtype=torch.float32).unsqueeze(0)  # (1, obs_dim)
                obs_tensor = torch.tensor(obs[agent], dtype=torch.float32)
                # obs_tensor = torch.tensor(obs[agent], dtype=torch.float32)
                action, log_prob = mappo_agent.actor.act(obs_tensor)  # Get action and log_prob from actor
                # action, log_prob = mappo_agent.actor.act(obs[agent])  # Get action and log_prob from actor
                actions[agent] = int(action)
                log_probs[agent] = log_prob

            # Step the environment with the chosen actions
            next_obs, rewards, terminations, truncs, infos = base_env.step_pickup_drop(actions)

            # if agent in next_obs:
            #     next_obs_array = next_obs[agent]
            # else:
            #     next_obs_array = torch.tensor(0.0)

            # if isinstance(next_obs_array, np.ndarray):
            #     next_obs_tensor = torch.tensor(next_obs_array, dtype=torch.float32)

            # if next_obs_tensor.dim() == 1:
            #     next_obs_tensor = next_obs_tensor.unsqueeze(0)  # Add batch dim

            # next_value = mappo_agent.critic(next_obs_tensor)


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
                    # 'action': actions[agent],
                    'action': action_roll,
                    # 'log_prob': log_probs[agent],
                    'log_prob': log_prob_roll,
                    'reward': rewards[agent],
                    'termination': terminations[agent],
                    # 'next_state': next_obs[agent],
                    'next_state': next_state,
                    'next_value': next_value,
                    # 'next_value': mappo_agent.critic(next_obs[agent]),
                })


                # if episode_rewards[agent] + rewards[agent] <= 3.0:  ## TODO: check this, sum is more than 3.0 without this check
                episode_rewards[agent] += rewards[agent]  # Accumulate rewards

            obs = next_obs
            # print(terminations)
            done_flags.update(terminations)
            # terminations = any(terminations.values())  # Episode ends if any agent is done

            # Update the model at specified intervals
            if len(rollouts) >= batch_size:
                print('aaaa')
                mappo_agent.update(rollouts)  # Update the model using the rollouts
                rollouts = []  # Clear rollouts buffer for the next batch

        print(f"Episode {episode + 1}/{num_episodes}: Rewards = {episode_rewards}")
        print(terminations)
        cumulative_rewards.append(episode_reward_single)
        per_agent_rewards_all.append(episode_rewards.copy())  # Store a copy!
        # Optional: Save model checkpoints or logging
        # if episode % checkpoint_interval == 0:
        #     torch.save(mappo_agent.actor.state_dict(), 'actor_checkpoint.pth')
        #     torch.save(mappo_agent.critic.state_dict(), 'critic_checkpoint.pth')
    plot_rewards(cumulative_rewards, per_agent_rewards_all)

main()
