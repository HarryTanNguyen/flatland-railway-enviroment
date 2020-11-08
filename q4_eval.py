# This file is used to evaluate the prioritized planning and genetic algorithm for task 4


import time
import threading

import numpy as np
import pyglet
from collections import deque
from flatland.utils.rendertools import RenderTool
from flatland.utils.graphics_pgl import RailViewWindow

from utils.environment_utils import create_multi_agent_environment
from q1_priority_plan import search,prioritized_planning,genetic_algorithm

# Evaluates the A* search algorithm over a number of samples.
def evalfun(num_samples = 100, timed=True, debug = False, refresh = 0.1):

    # A list of (mapsize, agent count) tuples, change or extend this to test different sizes.
    #problemsizes = [(5, 3), (7, 4), (9, 5), (11, 6), (13, 7)]
    problemsizes = [(7, 4)]

    # Create a list of seeds to consider.
    #seeds = numpy.random.randint(2**29, size=3*num_samples)
    scores = []
    successes = 0
    completion_window = deque(maxlen=100)
    completion = []
    runtime=[]
    schedule_lenth =[]
    seeds = 37429879
    print("%10s\t%8s\t%8s\t%9s" % ("Dimensions", "Success", "Rewards", "Runtime"))
    for problemsize in problemsizes:
        # Create environments while they are not the intended dimension.
        j = 0
        env = create_multi_agent_environment(problemsize[0], problemsize[1], timed, seeds)
        for _ in range(0, num_samples):

            # Create environments while they are not the intended dimension.

            # Create a renderer only if in debug mode.
            if debug:
                env_renderer = RenderTool(env, screen_width=1920, screen_height=1080)


            # Find the schedules
            start = time.time()
            _, schedule = genetic_algorithm(env)
            duration = time.time() - start;
            runtime.append(duration)
            schedule_lenth.append(len(schedule))
            if debug:
                env_renderer.render_env(show=True, frames=False, show_observations=False)
                time.sleep(refresh)

            # Validate that environment state is unchanged.
            #assert env.num_resets == 1 and env._elapsed_steps == 0

            # Run the schedule
            success = False
            sumreward = 0
            for action in schedule:
                _, _reward_dict, _done, _ = env.step(action)
                success = all(_done.values())
                sumreward = sumreward + sum(_reward_dict.values())
                if debug:
                    env_renderer.render_env(show=True, frames=False, show_observations=False)
                    time.sleep(refresh)
            # Print the performance of the algorithm
            if success==True:
                successes += 1

            tasks_finished = np.sum([int(_done[idx]) for idx in range(env.get_num_agents())])
            completion_window.append(tasks_finished / max(1, env.get_num_agents()))
            completion.append((np.mean(completion_window)))
            scores.append(sumreward)
            print("%10s\t%8s\t%8.3f\t%9.6f" % (str(problemsize), str(success), sumreward, duration))
            print(schedule_lenth)
            env.reset()
        print("Number of sucesses", successes)
        print("Number of samples", num_samples)
        print("Successful:    %8.2f%%" % (100 * successes / num_samples))
        print("Mean reward:   %8.2f" % (np.mean(scores)))
        print("Median reward: %8.2f" % (np.median(scores)))
        print("Instances solved: %8.2f" % (np.mean(completion)))
        print("Run Time   %8.2f" % (np.mean(runtime)))
        print("Avg schedule length %8.2f", np.mean(schedule_lenth))

if __name__ == "__main__":

    # Number of maps of each size to consider.
    _num_maps = 100
    # If _timed = true, impose release dates and deadlines. False for regular (Assignment 1) behavior.
    _timed = True

    _debug = False
    _refresh = 0.3

    if (_debug):
        window = RailViewWindow()

    evalthread = threading.Thread(target=evalfun, args=(_num_maps,_timed,_debug,_refresh,))
    evalthread.start()

    if (_debug):
        pyglet.clock.schedule_interval(window.update_texture, 1/120.0)
        pyglet.app.run()

    evalthread.join()
