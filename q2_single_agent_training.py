from argparse import Namespace
from collections import deque

#base_dir = Path(__file__).resolve().parent.parent
#sys.path.append(str(base_dir))

from reinforcement_learning.dddqn_policy import DDDQNPolicy
import numpy as np
import torch

from utils.observation_utils import normalize_observation
from utils.environment_utils import create_default_single_agent_environment

"""
This file shows how to train a single agent using a reinforcement learning approach.
Documentation: https://flatland.aicrowd.com/getting-started/rl/single-agent.html

This is a simple method used for demonstration purposes.
multi_agent_training.py is a better starting point to train your own solution!
"""
def train_agent(seed, n_episodes, timed = False):

    # Observation parameters
    observation_radius = 10

    # Exploration parameters
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.997  # for 2500ts

    # Setup the environment
    env, max_steps, x_dim, y_dim, observation_tree_depth, _ = create_default_single_agent_environment(seed, timed)
    env.reset(True, True)

    # Calculate the state size given the depth of the tree observation and the number of features
    n_features_per_node = env.obs_builder.observation_dim
    n_nodes = 0
    for i in range(observation_tree_depth + 1):
        n_nodes += np.power(4, i)
    state_size = n_features_per_node * n_nodes

    # The action space of flatland is 5 discrete actions
    action_size = 5
    action_dict = dict()

    # And some variables to keep track of the progress
    scores_window = deque(maxlen=100)  # todo smooth when rendering instead
    completion_window = deque(maxlen=100)
    scores = []
    completion = []
    action_count = [0] * action_size
    agent_obs = [None] * env.get_num_agents()
    agent_prev_obs = [None] * env.get_num_agents()
    agent_prev_action = [2] * env.get_num_agents()
    update_values = False

    # Training parameters
    training_parameters = {
        'buffer_size': int(1e5),
        'batch_size': 32,
        'update_every': 8,
        'learning_rate': 0.5e-4,
        'tau': 1e-3,
        'gamma': 0.99,
        'buffer_min_size': 0,
        'hidden_size': 256,
        'use_gpu': False
    }

    # Double Dueling DQN policy
    policy = DDDQNPolicy(state_size, action_size, Namespace(**training_parameters), seed=(seed+1))

    for episode_idx in range(n_episodes):
        score = 0

        # Reset environment
        obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)

        # Build agent specific observations
        for agent in env.get_agent_handles():
            if obs[agent]:
                agent_obs[agent] = normalize_observation(obs[agent], observation_tree_depth, observation_radius=observation_radius)
                agent_prev_obs[agent] = agent_obs[agent].copy()

        # Run episode
        for _ in range(max_steps - 1):
            for agent in env.get_agent_handles():
                if info['action_required'][agent]:
                    # If an action is required, we want to store the obs at that step as well as the action
                    update_values = True
                    action = policy.act(agent_obs[agent], eps=eps_start)
                    action_count[action] += 1
                else:
                    update_values = False
                    action = 0
                action_dict.update({agent: action})

            # Environment step
            next_obs, all_rewards, done, info = env.step(action_dict)

            # Update replay buffer and train agent
            for agent in range(env.get_num_agents()):
                # Only update the values when we are done or when an action was taken and thus relevant information is present
                if update_values or done[agent]:
                    policy.step(agent_prev_obs[agent], agent_prev_action[agent], all_rewards[agent], agent_obs[agent], done[agent])

                    agent_prev_obs[agent] = agent_obs[agent].copy()
                    agent_prev_action[agent] = action_dict[agent]

                if next_obs[agent]:
                    agent_obs[agent] = normalize_observation(next_obs[agent], observation_tree_depth, observation_radius=observation_radius)

                score += all_rewards[agent]

            if done['__all__']:
                break

        # Epsilon decay
        eps_start = max(eps_end, eps_decay * eps_start)

        # Collection information about training
        tasks_finished = np.sum([int(done[idx]) for idx in env.get_agent_handles()])
        completion_window.append(tasks_finished / max(1, env.get_num_agents()))
        scores_window.append(score / (max_steps * env.get_num_agents()))
        completion.append((np.mean(completion_window)))
        scores.append(np.mean(scores_window))
        action_probs = action_count / np.sum(action_count)

        if episode_idx % 100 == 0:
            end = "\n"
            action_count = [1] * action_size
        else:
            end = " "

        print('\rTraining {} agents on {}x{}\t Episode {}\t Average Score: {:.3f}\tDones: {:.2f}%\tEpsilon: {:.2f} \t Action Probabilities: \t {}'.format(
            env.get_num_agents(),
            x_dim, y_dim,
            episode_idx,
            np.mean(scores_window),
            100 * np.mean(completion_window),
            eps_start,
            action_probs
        ), end=end)

    # Return trained policy.
    return policy

def train_policy(seed, n_episodes, timed, filename):
    policy = train_agent(seed, n_episodes, timed)
    torch.save(policy.qnetwork_local, filename)

if __name__ == "__main__":

    seed = 42
    timed = True
    n_episodes = 500
    filename = "./rl-weights-withtimed.pth"

    train_policy(seed, n_episodes, timed, filename)
