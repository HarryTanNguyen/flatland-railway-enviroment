import random
import numpy as np

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator, complex_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator, complex_schedule_generator
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.observations import TreeObsForRailEnv, TreeObsForRailEnvExtended
from math import floor

def create_default_single_agent_environment(seed, timed):
    
    # Default observation parameters
    observation_tree_depth = 2
    observation_max_path_depth = 30

    # Default (tree) observation builder
    predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
    tree_observation = TreeObsForRailEnvExtended(max_depth=observation_tree_depth, predictor=predictor)

    # Unpack the return values of the default environment, in order to re-pack them for our return value.
    env, max_steps, x_dim, y_dim = _create_default_single_agent_environment(seed, timed, tree_observation)

    return env, max_steps, x_dim, y_dim, observation_tree_depth, observation_max_path_depth

def _create_default_single_agent_environment(seed, timed, observation_builder):
    
    # Environment parameters
    x_dim = 25
    y_dim = 25
    n_agents = 1
    n_cities = 4
    max_rails_between_cities = 2
    max_rails_in_city = 3

    return _create_single_agent_environment(seed, x_dim, y_dim, n_agents, n_cities, timed, \
                                           max_rails_between_cities, max_rails_in_city, \
                                           observation_builder)


def _create_single_agent_environment(seed, x_dim, y_dim, n_agents, n_cities, timed,
                                    max_rails_between_cities, max_rails_in_city,
                                    observation_builder):

    # Set the seeds
    random.seed(seed)
    np.random.seed(seed)

    # Setup the environment
    env = RailEnv(
        width=x_dim,
        height=y_dim,
        rail_generator=sparse_rail_generator(
            max_num_cities=n_cities,
            seed=seed,
            grid_mode=False,
            max_rails_between_cities=max_rails_between_cities,
            max_rails_in_city=max_rails_in_city
        ),
        schedule_generator=sparse_schedule_generator(timed=timed),
        number_of_agents=n_agents,
        obs_builder_object=observation_builder,
        random_seed=seed
    )

    # Compute the maximum number of steps allowed.
    max_steps = int(4 * 2 * (env.height + env.width + (n_agents / n_cities)))

    # Return produced environment
    return env, max_steps, x_dim, y_dim


def create_multi_agent_environment(dimension, num_agents, timed, seed):
    # Create new environment.

    env = RailEnv(
                width=dimension,
                height=dimension,
                rail_generator=complex_rail_generator(
                                        nr_start_goal=int(1.5 * num_agents),
                                        nr_extra=int(1.2 * num_agents),
                                        min_dist=int(floor(dimension / 2)),
                                        max_dist=99999,
                                        seed=0),
                schedule_generator=complex_schedule_generator(timed=timed),
                malfunction_generator_and_process_data=None,
                number_of_agents=num_agents)

    env.reset(random_seed=int(seed))

    return env

def create_multi_agent_rail_env(seed,timed):
    n_agents = 4
    # Environment parameters
    x_dim = 25
    y_dim = 25
    n_cities = 4
    max_rails_between_cities = 2
    max_rails_in_city = 3

    # Default observation parameters
    observation_tree_depth = 2
    observation_max_path_depth = 30

    # Default (tree) observation builder
    predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
    tree_observation = TreeObsForRailEnvExtended(max_depth=observation_tree_depth, predictor=predictor)

    random.seed(seed)
    np.random.seed(seed)

    env= RailEnv(
        width=x_dim, height=y_dim,
        rail_generator=sparse_rail_generator(
            max_num_cities=n_cities,
            seed=seed,
            grid_mode=False,
            max_rails_between_cities=max_rails_between_cities,
            max_rails_in_city=max_rails_in_city
        ),
        schedule_generator=sparse_schedule_generator(timed=timed),
        number_of_agents=n_agents,
        malfunction_generator_and_process_data=None,
        obs_builder_object=tree_observation,
        random_seed=seed
    )
    max_steps = int(4 * 2 * (env.height + env.width + (n_agents / n_cities)))
    return env, max_steps, x_dim, y_dim, observation_tree_depth, observation_max_path_depth
