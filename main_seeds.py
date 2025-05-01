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

def plot_rewards(cumulative_rewards, num_episodes):
    all_rewards = np.array(cumulative_rewards)

    mean_reward = np.mean(all_rewards, axis=0)
    std_reward = np.std(all_rewards)

    plt.plot(mean_reward, label='Mean Reward')
    plt.fill_between(np.arange(num_episodes), mean_reward - std_reward, mean_reward + std_reward, alpha=0.2, label='Standard Deviation')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Learning Curve Across Seeds')
    plt.legend()
    plt.savefig('Mean and Std Dev Rewards Diff Seeds.png') 
    plt.close()  

def main():

    # per_agent_rewards_all = []  # Store per-agent rewards per episode
    seeds = [42, 162, 120, 14, 45]
    num_episodes = 500
    batch_size = 64
    for s in seeds:
        # Initialize the environment (GoalBasedSimpleSpread)
        cumulative_rewards = []  # Total reward across all agents, per episode
        base_env = PickUpDropOffSimpleSpread(seed=s, max_cycles=30, num_tasks=1)  # Pass the number of tasks here (1 pickup/dropoff pair per agent)
        agent = base_env.agents[0]  # Just pick one agent
        obs_dim = base_env.observation_spaces(agent).shape[0]
        act_dim = base_env.action_spaces(agent).n

        # print(obs_dim)
        # print(act_dim)
        actor = Actor(obs_dim=obs_dim, act_dim=act_dim)
        critic = Critic(obs_dim=obs_dim)
        mappo_agent = MAPPO(base_env, actor, critic)

        # Training parameters

        
        # Should move this to MAPPO class
        for episode in range(num_episodes):
            obs = base_env.reset()  # Reset the environment and get initial observations
            done = False
            episode_rewards = {agent: 0 for agent in base_env.agents}  # Initialize rewards
            rollouts = []  # Store rollouts for updating the model
            

            print(base_env.pickups)
            print(base_env.dropoffs)
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

                for agent in base_env.agents:
                    if terminations[agent] and episode_rewards[agent] == 1 and infos[agent]['color'] == 'green':
                        episode_rewards[agent] += rewards[agent]

                # Store experiences in the rollouts buffer
                for agent in base_env.agents:
                    # if terminations[agent] and infos[agent]['color'] == 'green':
                    #     episode_rewards[agent] += rewards[agent]
                    #     continue
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
                    # {'agent_0': {'color': 'red'}, 'agent_1': {'color': 'orange'}, 'agent_2': {'color': 'orange'}}
                    # if episode_rewards[agent] + rewards[agent] <= 3.0:  ## TODO: check this, sum is more than 3.0 without this check

                    if infos[agent]['color'] == 'orange' and episode_rewards[agent] == 0:
                        episode_rewards[agent] += rewards[agent]  
                    
                    # if infos[agent]['color'] == 'green':
                        # print('avvd')
                        # episode_rewards[agent] += rewards[agent]  

                obs = next_obs
                done_flags.update(terminations)
                # terminations = any(terminations.values())  # Episode ends if any agent is done

                # Update the model at specified intervals
                if len(rollouts) >= batch_size:
                    # print('updated using rollouts')
                    mappo_agent.update_mappo(rollouts)  # Update the model using the rollouts
                    rollouts = []  # Clear rollouts buffer for the next batch

            print(f"Episode {episode + 1}/{num_episodes}: Rewards = {episode_rewards}")
            # print(terminations)
            cumulative_rewards.append(sum(episode_rewards.values()))
            # per_agent_rewards_all.append(episode_rewards.copy())  # Store a copy!



    plot_rewards(cumulative_rewards, num_episodes)

main()
