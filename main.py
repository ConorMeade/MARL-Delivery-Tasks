import torch
import numpy as np
from PickUpDroppOffSimpleSpread import PickUpDropOffSimpleSpread
from mappo import MAPPO
from actor_critic import Actor, Critic

def main():
    # Initialize the environment (GoalBasedSimpleSpread)
    base_env = PickUpDropOffSimpleSpread(seed=42, num_tasks=1)  # Pass the number of tasks here (1 pickup/dropoff pair per agent)

    # print(base_env.agents)

    # print(base_env.observation_spaces(0))
    # Define observation and action dimensions
    obs_dim = base_env.observation_spaces(base_env.agents[0]).shape[0]  # Size of observation space for an agent
    act_dim = base_env.action_spaces(base_env.agents[0]).n  # Size of action space for an agent

    print(obs_dim)
    print(act_dim)
    # Instantiate Actor and Critic separately
    actor = Actor(obs_dim=obs_dim, act_dim=act_dim)  # Instantiate the Actor
    critic = Critic(obs_dim=obs_dim)  # Instantiate the Critic

    # Initialize MAPPO model
    mappo_agent = MAPPO(base_env, actor, critic)

    # Training parameters
    num_episodes = 1000
    batch_size = 32
    update_interval = 10  # Update the model after this many steps

    for episode in range(num_episodes):
        obs = base_env.reset()  # Reset the environment and get initial observations
        done = False
        episode_rewards = {agent: 0 for agent in base_env.agents}  # Initialize rewards
        rollouts = []  # Store rollouts for updating the model
        
        # Start episode loop
        while not done:
            actions = {}
            log_probs = {}

            # Collect actions and log_probs for each agent
            for agent in base_env.agents:
                action, log_prob = mappo_agent.actor.act(obs[agent])  # Get action and log_prob from actor
                actions[agent] = action
                log_probs[agent] = log_prob

            # Step the environment with the chosen actions
            next_obs, rewards, dones, truncs, infos = base_env.step(actions)

            # Store experiences in the rollouts buffer
            for agent in base_env.agents:
                rollouts.append({
                    'state': obs[agent],
                    'action': actions[agent],
                    'log_prob': log_probs[agent],
                    'reward': rewards[agent],
                    'done': dones[agent],
                    'next_state': next_obs[agent],
                    'next_value': mappo_agent.critic(next_obs[agent]),
                })

                episode_rewards[agent] += rewards[agent]  # Accumulate rewards

            obs = next_obs
            done = any(dones.values())  # Episode ends if any agent is done

            # Update the model at specified intervals
            if len(rollouts) >= batch_size:
                mappo_agent.update(rollouts)  # Update the model using the rollouts
                rollouts = []  # Clear rollouts buffer for the next batch

        print(f"Episode {episode + 1}/{num_episodes}: Rewards = {episode_rewards}")

        # Optional: Save model checkpoints or logging
        # if episode % checkpoint_interval == 0:
        #     torch.save(mappo_agent.actor.state_dict(), 'actor_checkpoint.pth')
        #     torch.save(mappo_agent.critic.state_dict(), 'critic_checkpoint.pth')

main()
