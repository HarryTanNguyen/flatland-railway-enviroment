import time
import threading
import pyglet
from argparse import Namespace

from reinforcement_learning.dddqn_policy import DDDQNPolicy
import numpy as np
import torch

from flatland.utils.rendertools import RenderTool
from flatland.utils.graphics_pgl import RailViewWindow
from utils.observation_utils import normalize_observation
from utils.environment_utils import create_default_single_agent_environment

def load_policy(filename, state_size=231, hidden_layer_size=256, seed=None):

    # Training parameters
    training_parameters = {
        'buffer_size': int(1e5),
        'batch_size': 32,
        'update_every': 8,
        'learning_rate': 0.5e-4,
        'tau': 1e-3,
        'gamma': 0.99,
        'buffer_min_size': 0,
        'hidden_size': hidden_layer_size,
        'use_gpu': False
    }

    # The action space of flatland is 5 discrete actions
    action_size = 5

    # Create Double DQN Policy object by loading the network weights from file.
    policy = DDDQNPolicy(state_size, action_size, Namespace(**training_parameters), seed=seed)
    policy.qnetwork_local = torch.load(filename)

    return policy

def evaluate(seed=37429879, timed=False, filename="./rl-weights.pth", debug=False, refresh=1):

    # Attempt to load policy from disk.
    policy = load_policy(filename, seed=seed)

    # Create environment with given seeding.
    env, max_steps, _, _, observation_tree_depth, _ = create_default_single_agent_environment(seed+1, timed)

    # Fixed environment parameters (note, these must correspond with the training parameters!)
    observation_radius = 10

    env_renderer = None
    if (debug):
        env_renderer = RenderTool(env, screen_width=1920, screen_height=1080)

    # Create container for the agent actions and observations.
    action_dict = dict()
    agent_obs = [None] * env.number_of_agents

    num_maps = 100
    scores = []
    successes = 0

    for _ in range(0, num_maps):

        # Create a new map.
        obs, info = env.reset(True, True)
        score = 0
    
        if debug:
            env_renderer.reset()
            env_renderer.render_env(show=True, frames=False, show_observations=False)
            time.sleep(refresh)

        # Run episode
        for _ in range(max_steps - 1):

            # Build agent specific observations
            for agent in env.get_agent_handles():
                if obs[agent]:
                    agent_obs[agent] = normalize_observation(obs[agent], observation_tree_depth, observation_radius=observation_radius)

            # If an action is required, select the action.
            for agent in env.get_agent_handles():
                action = 0
                if info['action_required'][agent]:
                    action = policy.act(agent_obs[agent], eps=0.08)
                    #print("Required " + str(action))
                action_dict.update({agent: action})

            # Environment step
            obs, all_rewards, done, info = env.step(action_dict)
    
            if debug:
                env_renderer.render_env(show=True, frames=False, show_observations=False)
                time.sleep(refresh)

            # Track rewards.
            score = score + all_rewards[agent]

            if done[agent]:
                successes = successes + 1
                break

        # Record scores.
        scores.append(score)

    print("Successful:    %8.2f%%" % (100 * successes / num_maps))
    print("Mean reward:   %8.2f"  % (np.mean(scores)))
    print("Median reward: %8.2f"  % (np.median(scores)))

def main():

    #seed = 37429879
    seed = 32617879
    timed = True
    filename = "./rl-weights-withtimed.pth"
    _debug = False
    _refresh = 0.05

    if (_debug):
        window = RailViewWindow()

    evalthread = threading.Thread(target=evaluate, args=(seed, timed, filename, _debug, _refresh,))
    evalthread.start()

    if (_debug):
        pyglet.clock.schedule_interval(window.update_texture, 1/120.0)
        pyglet.app.run()

    evalthread.join()

if __name__ == "__main__":
    main()
