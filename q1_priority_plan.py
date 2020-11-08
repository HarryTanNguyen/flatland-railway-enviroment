import numpy as np
import itertools
import math
import heapq
import random
from dataclasses import dataclass, field
from typing import Any
from copy import deepcopy
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import RailAgentStatus

class StateConverter:
    """
    Used to convert position/tile to state and vice versa
    Encode state to a number
    """
    def __init__(self, env: RailEnv):
        self.width = env.rail.width
        # Number of tiles
        self.num_tiles = env.rail.width * env.rail.height
        # number of states = Number of tiles * number of direction (4)
        self.num_states = 4 * self.num_tiles

    def position_to_state(self, row, col, dir):
        """
        Convert current position to state
        Example: Current agent has position at (2,3) current direction is South (2) in the RailW having width
                of 5 --> state = 2 + 4*3 + 4*5*2 = 54
        """
        return dir + 4 * col + 4 * self.width * row

    def position_to_tile(self, position):
        """
        Convert position (x_axis,y_axis) to tile number
        Example: position (0,1) with the width (col_size = 3) ==> tile number = 1
        """
        return position[1] + self.width * position[0]

    def state_to_position(self, state):
        """
        Convert state (int) to position (int,int) and direction (int)
        Example: State = 54 - width :5
        dir = 54 % 4 = 2 (South)
        col_pos = ((54-2)/4) % 5 = 12 % 5 = 3
        row_pos = ((54-2-3*4))/(4*5) = 40/20 = 2
        """
        dir = state % 4
        col = ((state - dir) / 4) % self.width
        row = (state - dir - col * 4) / (4 * self.width)
        return row, col, dir

    @staticmethod
    def state_to_tile(state):
        return np.int32((state - state % 4) / 4)


def convert_to_transition(env: RailEnv, conv: StateConverter):
    """
    This function is used to fill the transition and valid_action table
    The transition for finding the new state after doing a specific action in specific state
    The valid_action table for checking if doing action in current state is valid
    """

    # Transition is a function: [state][action] -> new state
    # Transition will be 2D Matrix initially fill with 1 with
    #           - size of row = number of states
    #           - size of column = number of action (default = 5)
    # If the train in state i and it takes action j then it will get the new state at transition[state=i][action=j]
    transition = -np.ones((conv.num_states, 5), dtype=np.int32)

    # Action is valid in a particular state if it leads to a new position.
    # There are five actions so size of column will be 5

    # Example: if a train in state i and want to do action j
    # then it needs to check if the valid_action[i][j] (if valid_action[i][j]==1 the train can do the action)
    valid_action = np.zeros((conv.num_states, 5), dtype=np.int32)

    # Compute the valid_action and transition tables
    for row in range(0, env.rail.height):
        for col in range(0, env.rail.width):
            for dir in range(0, 4):
                # For each direction, each col and each row

                # Convert the current position to state
                state = conv.position_to_state(row, col, dir)

                # Compute the number of possible transitions.
                # First we get the possible transition for the current state.
                # The "get_transitions" function returns a tuple (0 or 1,0 or 1,0 or 1,0 or 1). It shows us which
                # direction agent can take
                possible_transitions = env.rail.get_transitions(row, col, dir)
                # Count the number of direction we can go in current state
                num_transitions = np.count_nonzero(possible_transitions)

                if num_transitions > 0:

                    # The easy case: stop moving holds current state. (Agent can stop moving at every state

                    # Since the action is stop moving, after doing this action
                    # the new state will be the same as the old state
                    transition[state][RailEnvActions.STOP_MOVING] = state
                    # Set to valid action (=1)
                    valid_action[state][RailEnvActions.STOP_MOVING] = 1

                    # Forward is only possible in two cases, there is only 1 option mean there is only one rail, or
                    # the current direction can be maintained. Stop otherwise.

                    # There is only one option for agent
                    if num_transitions == 1:
                        # Get the index where possible_transition == 1 , the index implies the new direction after agent
                        # continue moving forward
                        new_direction = np.argmax(possible_transitions)
                        # Calculate new position when apply the only transition we just found
                        new_position = get_new_position((row, col), new_direction)
                        # Calculate new state
                        transition[state][RailEnvActions.MOVE_FORWARD] = conv.position_to_state(new_position[0],
                                                                                                new_position[1],
                                                                                                new_direction)
                        # Set to valid_action
                        valid_action[state][RailEnvActions.MOVE_FORWARD] = 1

                    # If there are more than one transition and one of the transition has the same direction as the
                    # agent is facing
                    elif possible_transitions[dir] == 1:
                        new_position = get_new_position((row, col), dir)
                        transition[state][RailEnvActions.MOVE_FORWARD] = conv.position_to_state(new_position[0],
                                                                                                new_position[1], dir)
                        valid_action[state][RailEnvActions.MOVE_FORWARD] = 1
                    # If there is no option, the agent will stay
                    else:
                        transition[state][RailEnvActions.MOVE_FORWARD] = state

                    # Left is only possible if there is a transition out to the left of
                    # the current direction. Otherwise, we move like we would if going Forward.
                    new_direction = (dir - 1) % 4
                    # If there is a transition out to the left of the current direction
                    if possible_transitions[new_direction]:
                        new_position = get_new_position((row, col), new_direction)
                        transition[state][RailEnvActions.MOVE_LEFT] = conv.position_to_state(new_position[0],
                                                                                             new_position[1],
                                                                                             new_direction)
                        # If moving the left and moving forward are not end up in the same state, set Move_left action
                        # = 1
                        valid_action[state][RailEnvActions.MOVE_LEFT] = transition[state][RailEnvActions.MOVE_LEFT] != \
                                                                        transition[state][RailEnvActions.MOVE_FORWARD]
                    else:
                        transition[state][RailEnvActions.MOVE_LEFT] = transition[state][RailEnvActions.MOVE_FORWARD]

                    # Right is only possible if there is a transition out to the Right of
                    # the current direction. Otherwise, we move like we would if going
                    # Forward.
                    new_direction = (dir + 1) % 4
                    if possible_transitions[new_direction]:
                        new_position = get_new_position((row, col), new_direction)
                        transition[state][RailEnvActions.MOVE_RIGHT] = conv.position_to_state(new_position[0],
                                                                                              new_position[1],
                                                                                              new_direction)
                        valid_action[state][RailEnvActions.MOVE_RIGHT] = transition[state][RailEnvActions.MOVE_RIGHT] != \
                                                                         transition[state][RailEnvActions.MOVE_FORWARD]
                    else:
                        transition[state][RailEnvActions.MOVE_RIGHT] = transition[state][RailEnvActions.MOVE_FORWARD]

    return (transition, valid_action)


def all_pairs_shortest_paths(num_states, transition):
    """
    Calculate the distance table from one state to another state
    """

    dist = np.ones((num_states, num_states), dtype=np.int32) * np.inf

    # Initialize;
    # neighbors of the current state are at distance 1 step, current state at 0 steps.
    for state in range(0, num_states):
        # If agent MOVE FORWARD, MOVE LEFT ,MOVE RIGHT
        for action in range(1, 4):
            # Get the new state after doing action
            next_state = transition[state][action]
            if next_state != -1 and next_state != state:
                dist[state][next_state] = 1
        # dist after STOP MOVING
        dist[state][state] = 0

    # FLoyd-Warshall algorithm to compute distances of shortest paths.
    for k in range(0, num_states):
        for i in range(0, num_states):
            for j in range(0, num_states):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist


class SearchState:
    def __init__(self, positions, actives):
        """
        Search state contains the positions of the agents (the positions are encoded into numbers (state) (using position_to_state
        function in StateConverter ))
        """
        self.positions = positions
        self.actives = actives
        self.hash = hash(self.actives.tobytes()) + 31 * hash(self.positions.tobytes())

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.actives == other.actives and np.array_equal(self.positions, other.positions)
        else:
            return NotImplemented


@dataclass(order=True)
class SearchNode:

    f: int
    neg_g: int

    parent: Any = field(compare=False)
    action: Any = field(compare=False)
    searchenv: Any = field(compare=False)
    searchstate: Any = field(compare=False)
    time_step: Any = field(compare= False)

    def __init__(self, neg_g, parent, action, searchenv, searchstate,time_step = 0):

        self.parent = parent
        self.action = action

        # searchenv contains the valid_action, transition, shortest distance table and goal state
        # searchstate contains State Converted, Initial state, Transition and valid action table
        self.searchenv = searchenv
        self.searchstate = searchstate

        self.time_step = time_step
        # g value = the distance from the root node to the current node
        # the reason we use negative g is to maintain the intended sorting of the A* search priority queue (heapq)
        self.neg_g = neg_g
        # f = the distance from the root node to the current node + the shortest path from current state to goal state
        self.f = self.get_evaluation()

    def __hash__(self):
        return self.searchstate.__hash__()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.searchstate.__eq__(other.searchstate) and self.time_step == other.time_step:
                return True
            else:
                return False
        else:
            return NotImplemented

    def agents_at_goal(self):
        """
        Check if the agents are at one of the goal states.
        """
        return self.searchenv.conv.state_to_tile(self.searchstate.positions) == self.searchenv.goal_tile

    def is_goal_state(self):
        """
        Check if all agents are at the goal position --> goal state

        """
        return self.agents_at_goal().all()

    def get_evaluation(self):
        #print("Penalty apply: ",self.get_timing_penalty(self.time_step, self.searchenv.env._max_episode_steps))
        return (-self.neg_g + self.get_timing_penalty(self.time_step, self.searchenv.env._max_episode_steps)) + self.get_heuristic()

    def get_heuristic(self):
        # Compute the shortest paths from current state (position of all agent to their goal)
        shortest_to_goal_states = self.searchenv.shortest[self.searchstate.positions.reshape(len(self.searchstate.positions),1), self.searchenv.goal_states]
        # There are many ways for a train to enter the goal position (4 direction) choose the shortest one
        shortest_to_goal_state = np.min(shortest_to_goal_states, 1)
        # heuristic value is the longest distance that need for all agent to reach the goal position
        h = np.int32(np.max(shortest_to_goal_state))
        # print("Distance to Goal",h)
        total_forward_penalty = 0
        if self.searchstate.actives == RailAgentStatus.ACTIVE:
            for i in range(1,h):
                total_forward_penalty += self.get_timing_penalty(self.time_step + i, self.searchenv.env._max_episode_steps)
        return h + total_forward_penalty

    def get_occupied_tiles(self):
        """
        Check if tile has been occupied by other agents
        """
        occupied = np.zeros(self.searchenv.conv.num_tiles)
        #Convert current state (positions of agents) to tile indices
        tiles = self.searchenv.conv.state_to_tile(self.searchstate.positions)
        valid_tiles = tiles[self.searchstate.actives == 1]
        occupied[valid_tiles] = 1
        return occupied

    def get_timing_penalty(self, elapsed_steps, max_episode_steps):

        # By default, no penalty.
        penalty = 0

        # Penalty only applies if the agent is active.
        if self.searchstate.actives == RailAgentStatus.ACTIVE:

            # Compute the number of steps the agent is outside bounds.
            steps_outside = 0

            if elapsed_steps < self.searchenv.release:
                steps_outside = self.searchenv.release - elapsed_steps
            if elapsed_steps > self.searchenv.deadline:
                steps_outside = elapsed_steps - self.searchenv.deadline

            # Compute the normalized penalty.
            penalty = ((steps_outside * steps_outside) / (max_episode_steps * max_episode_steps / 4))

        return penalty

    def get_all_valid_actions(self):
        """
        Return a list containing IDs of the available action for each agent
        """

        # Select, for each agent, the valid actions based on its position (state).
        agent_actions = self.searchenv.valid_actions[self.searchstate.positions[0]]

        #print("Agent Action: ",agent_actions)

        # Mask the rail transition actions for idle agents.
        if self.searchstate.actives == 0:
            agent_actions = [0, 0, 1, 0, 1]     # STOP_MOVING, or MOVE_FORWARD.

        # Mask the rail transition actions for done agents.
        if self.agents_at_goal() == True:
            agent_actions = [1, 0, 0, 0, 0]     # DO_NOTHING only.

        # Identify for each agent the IDs of the valid actions (i.e., [0, 1, 1, 0, 0] --> [1, 2])
        agent_action_list =[]
        for i in range(len(agent_actions)):
            if agent_actions[i] == 1:
                agent_action_list.append(i)

        # Return list containing for each agent, the IDs of the actions available to it.
        return agent_action_list

    def expand_node(self, action,occupied_map):

        """
        Input:
         - actions: an array, where actions[agent] is the action id that agent id will try to take.
        """

        # Determine which tiles are occupied now.
        occupied = self.get_occupied_tiles()

        # Make copy the current search state (to modify).
        new_states = self.searchstate.positions.copy()
        new_actives = self.searchstate.actives.copy()
        occupied_table = occupied_map.copy()

        if len(occupied_table)-1 < self.time_step:
            for i in range (self.time_step - len(occupied_table) + 1):
                occupied_table.append(np.zeros(self.searchenv.conv.num_tiles))

        # Move agents in increasing order of their IDs.
        # for each agent
        for i in range(0, len(self.searchstate.positions)):

            # Get the current state.
            current_state = new_states[i]
            current_tile = self.searchenv.conv.state_to_tile(current_state)

            # Agent was inactive, wants to begin moving.
            if new_actives[i] == 0 and action == 2:
                if occupied_table[self.time_step][current_tile] == 1:
                    # Attempting to enter blocked tile, expand fails.
                    return None
                else:
                    # Activate agent, occupy tile.
                    new_actives[i] = 1
                    # occupied[current_tile] = 1

            # Agent was active, attempt to apply action
            elif new_actives[i] == 1:

                # The agent is trying to move, so it frees up the current tile.
                # occupied[current_tile] = 0
                # Compute the next state given the current state and action
                next_state = self.searchenv.transition[current_state, action]
                next_tile = self.searchenv.conv.state_to_tile(next_state)
                if occupied_table[self.time_step][next_tile] == 1:
                    # Attempting to enter blocked tile, expand fails.
                    return None
                else:
                    # free up the current tile
                    # occupied[current_tile] = 0
                    # Occupy the next tile
                    # occupied[next_tile] = 1
                    # Update the new state
                    new_states[i] = next_state

                    # Goal state reached, remove the occupancy, deactivate.
                    if next_tile == self.searchenv.goal_tile[i]:
                        # occupied[next_tile] = 0
                        new_actives[i] = 0
        # print("Action: ",action)
        # If the agent is not active and will remain inactive at next time step --> not incur step cost
        next_neg = self.neg_g
        if new_actives[0] == RailAgentStatus.ACTIVE:
            next_neg = self.neg_g - 1
        elif new_actives[i] == RailAgentStatus.READY_TO_DEPART:
            if self.time_step >= self.searchenv.release:
                next_neg = self.neg_g - 1
        return SearchNode(next_neg, self, action, self.searchenv, SearchState(new_states, new_actives),self.time_step + 1)

    def get_path(self):
        """

        Trace back from the goal node to root node to get actions and the states

        """
        action_dict = {self.searchenv.agent_id:self.action}
        if self.parent.parent is None:
            return [action_dict]
        else:
            path = self.parent.get_path()
            path.append(action_dict)
            return path

    def get_state_path(self):
        if self.parent is None:
            return [self.searchstate.positions[0]]
        else:
            path = self.parent.get_state_path()
            path.append(self.searchstate.positions[0])
            return path

    def get_time_step(self):
        if self.parent is None:
            return [self.time_step]
        else:
            time_step_list = self.parent.get_time_step()
            time_step_list.append(self.time_step)
            return time_step_list

class SearchEnv:

    def __init__(self, env: RailEnv,conv: StateConverter,model, shortest_path, agent_id):
        self.conv = conv
        self.transition = model[0]
        self.valid_actions = model[1]
        self.shortest = shortest_path
        # Initialized the starting state for all agent
        self.initial_state = np.zeros(1, dtype=np.int32)
        self.initial_active = np.zeros(1, dtype=np.int32)

        self.env = env
        # Compute the starting state of all agent
        self.agent_id = agent_id
        self.agent = env.agents[agent_id]
        self.initial_state[0] = self.conv.position_to_state(self.agent.initial_position[0], self.agent.initial_position[1], self.agent.initial_direction)
        self.release = self.agent.release_date
        self.deadline = self.agent.deadline
        # Compute the tile index for the target position of agents
        self.goal_tile = np.zeros(1, dtype=np.int32)
        self.goal_tile[0] = self.conv.position_to_tile(env.agents[agent_id].target)

        # Convert from tiles to states by adding directions 0 to 4.
        # numpy.mgrid returns a dense multi-dimensional "meshgrid". The dimension and number of the output arrays are
        # equal to the number of indexing dimension

        # when number of agents = 5
        # numpy.mgrid[0:len(env.agents),0:4][1] =array([[0, 1, 2, 3],
        #                                               [0, 1, 2, 3],
        #                                               [0, 1, 2, 3],
        #                                               [0, 1, 2, 3],
        #                                               [0, 1, 2, 3]])

        self.goal_states = np.mgrid[0:1,0:4][1] + self.goal_tile.reshape(1,1) * 4

    def get_root_node(self):
        initial_state = SearchState(self.initial_state.copy(), self.initial_active.copy())
        return SearchNode(0, None, None, self, initial_state)


def a_star_search(root, occupied):

    # Count the number of expansions and generations.
    expansions = 0
    generations = 0

    # Open list is a priority queue over search nodes, closed set is a hash-based set for tracking seen nodes.
    openlist = []
    closed = set({root})

    # Initially, open list is just the root node.
    heapq.heappush(openlist, root)

    # While we have candidates to expand,
    while len(openlist) > 0:

        # Get the highest priority search node.
        current = heapq.heappop(openlist)

        # Increment number of expansions.
        expansions = expansions + 1

        # If we expand the goal node, we are done.
        if current.is_goal_state():
            return (current.get_path(),current.get_state_path(),current.get_time_step(), expansions, generations)

        # Otherwise, we will generate all child nodes.
        # print("Valid Action: ", current.get_all_valid_actions())
        print("Current time step " +  str(current.time_step) +" Action: " + str(current.action))
        for action in current.get_all_valid_actions():
            print("Action: ",action)
            # Create successor node from action.
            nextnode = current.expand_node(action,occupied)

            # Generated one more node.
            generations = generations + 1

            # If this is a valid new node, append it to the open list.
            if nextnode is not None and not closed.__contains__(nextnode):
                print("Action Approved: " + str(action) +" F value of Next node: ", nextnode.f)
                closed.add(nextnode)
                heapq.heappush(openlist, nextnode)

    return (None,None, expansions, generations)


def compute_map(current_agent_id,agent_order,number_of_timestep,state_schedules, conv :StateConverter):
    """
    Compute the occupancy map for each agent

    Args:
        current_agent_id: The agent need to compute the occupancy map
        agent_order: contain a list of agent id that already computed schedule
        number_of_timestep: number of time step of other agent id
        state_schedules: schedule of other agents (contain the states of other agents)
        conv:

    Returns:
        occupancy_map: occupancy map of current agent id

    """
    #Find the agent has the highest number of time steps
    highest_timestep = 0
    # Find the highest time step
    if len(number_of_timestep) >0:
        highest_timestep = np.max(number_of_timestep)
    occupancy_map = []
    #   Since we don't know yet how many time step of the current id so
    # the number of time steps of the occupancy map == highest number of time step
    # of the current schedule
    for time_step in range(int(highest_timestep)):
        # Initialize the occupancy for current time step
        current_occupancy_map = np.zeros(conv.num_tiles)
        # We loop through schedule of each agent at current time step
        for i in range(len(state_schedules)):
            # Get the agent id of current schedule
            agent_of_schedule = agent_order[i]
            if time_step < len(state_schedules[i]):
                # The first case when the agent of current schedule is executed after the current agent
                if agent_of_schedule > current_agent_id:
                    # Get the current state
                    current_state = state_schedules[i][time_step]
                    # Convert the current state to tile index
                    current_tile = conv.state_to_tile(current_state)
                    # Occupied the current tile in the occupancy map
                    current_occupancy_map[current_tile] = 1
                    if time_step + 1 < len(state_schedules[i]):
                        # Get the next state
                        next_state = state_schedules[i][time_step + 1]
                        # Convert next state to next tile will be occupied
                        next_tile_index = conv.state_to_tile(next_state)
                        # Occupied the next tile in the occupancy map
                        current_occupancy_map[next_tile_index] = 1
                # The second case when the agent of current schedule is executed before the current agent
                else:
                    if time_step + 1 < len(state_schedules[i]):
                        # Get the next state
                        next_state = state_schedules[i][time_step + 1]
                        # Convert next state to next tile will be occupied
                        next_tile_index = conv.state_to_tile(next_state)
                        # Occupied the next tile in the occupancy map
                        current_occupancy_map[next_tile_index] = 1
                    if time_step + 2 < len(state_schedules[i]):
                        # Get the next 2 state
                        next_2state = state_schedules[i][time_step+2]
                        # Convert the current state to tile index
                        next_2tile = conv.state_to_tile(next_2state)
                        # Occupied the current tile in the occupancy map
                        current_occupancy_map[next_2tile] = 1
        occupancy_map.append(current_occupancy_map)
    return occupancy_map







#def search_stats(env:RailEnv):
    #return a_star_search(SearchEnv(env).get_root_node())


def prioritized_planning(env:RailEnv):
    """Implementation of prioritized planning with occupancy map"""
    schedules = []
    occupancy_map=[[] for i in range(len(env.agents))]

    n_timesteps = np.array([])
    state_schedule =[]
    conv = StateConverter(env)
    # Compute the transition and valid action table
    model = convert_to_transition(env, conv)

    # Calculate the shortest dist from one state to another state
    shortest = all_pairs_shortest_paths(conv.num_states, model[0])
    print("Done")
    l = list(range(len(env.agents)))
    # Create a random order
    random_order_agent = random.sample(l, len(l))
    print("Agent order: ",random_order_agent)


    for i in random_order_agent:
        # Compute occupancy map
        occupancy_map[i] = compute_map(i, random_order_agent, n_timesteps, state_schedule, conv)

        # Compute schedule,state for each agent based on the occupancy map
        each_schedule = a_star_search(SearchEnv(env,conv,model,shortest,i).get_root_node(),occupancy_map[i])
        schedules.append(each_schedule[0])
        state_schedule.append(each_schedule[1])
        n_timesteps = np.append(n_timesteps, [len(each_schedule[1])])


    # Combine separate actions into a list
    actions = combine(schedules,random_order_agent,int(np.max(n_timesteps)))

    return actions


def search_a_for_genetic(env:RailEnv,randomized):
    """
    prioritized planning with occupancy map used for genetic algorithm in q3
    """
    schedules = []
    occupancy_map=[[] for i in range(len(env.agents))]

    n_timesteps = np.array([])
    state_schedule =[]
    conv = StateConverter(env)
    # Compute the transition and valid action table
    model = convert_to_transition(env, conv)
    # Calculate the shortest dist from one state to another state
    shortest = all_pairs_shortest_paths(conv.num_states, model[0])
    random_order_agent = randomized
    print(random_order_agent)

    for i in random_order_agent:
        # Compute occupancy map
        occupancy_map[i] = compute_map(i, random_order_agent, n_timesteps, state_schedule, conv)

        # Compute schedule for each agent based on the occupancy map
        each_schedule = a_star_search(SearchEnv(env,conv,model,shortest,i).get_root_node(),occupancy_map[i])
        #print(each_schedule)
        schedules.append(each_schedule[0])
        state_schedule.append(each_schedule[1])
        n_timesteps = np.append(n_timesteps, [len(each_schedule[1])])

    # Combine separate actions into a list
    actions = combine(schedules,random_order_agent,int(np.max(n_timesteps)))

    return actions


def combine(schedules,agent_order_list,highest_timestep):
    """
    Combine list of actions from agents to one list

    """
    action=[]
    for timestep in range(highest_timestep-1):
        current_action = dict()
        index = 0
        for agent in agent_order_list:
            if timestep < len(schedules[index]):
                current_action[agent] = schedules[index][timestep][agent]
            else:
                current_action[agent] = 4
            index += 1
        action.append(current_action)
    return action





def search(env: RailEnv):
    # Creates a schedule of 8 steps of random actions.
    schedule = []
    for _ in range(0, 8):
        _actions = {}
        for i in env.get_agent_handles():
            _actions[i] = np.random.randint(0, 5)
        schedule.append(_actions)

    return schedule


def genetic_algorithm(env: RailEnv):
    """ Improvement of prioritized planning"""
    # Determine number of agent and number of combinations of randomized order
    n_agents = len(env.agents)
    n_combination = math.factorial(n_agents)
    agent_ids= list(range(n_agents))
    # Compute number of iteration for genetic algo
    # Run ten iterations
    n_iterations = 10

    # create an initial population. The population contains three combinations of randomized order
    population = []
    final_reward = dict()
    for i in range(3):
        randomized_order = random.sample(agent_ids, len(agent_ids))
        if not population.__contains__(randomized_order):
            population.append(randomized_order)

    highest_reward_of_each_iteration=[]
    action_plan = dict()
    # Run the genetic algorithm in n number of iteration
    for iteration in range(n_iterations):
        # Pass each randomized order to the modified a_star
        i=0
        reward = dict()
        print("*******************************************************New Population***********************************")
        print("Iteration ",iteration)
        # For each planning order in population we run it to get the cumulative reward
        for randomized_order in population:
            schedules = search_a_for_genetic(env,randomized_order)
            action_plan[i] = schedules
            test_env = deepcopy(env)
            success = False
            sumreward = 0
            #  apply to the copied environment to calculate the sumreward for fitness score
            for action in schedules:
                _, _reward_dict, _done, _ = test_env.step(action)
                success = all(_done.values())
                sumreward = sumreward + sum(_reward_dict.values())
            reward[i] = sumreward
            #print("Total Time Step: ",test_env._elapsed_steps)
            #print("Randomize order: ", randomized_order)
            #print("Sum reward: ",sumreward)
            i += 1
        final_reward = reward.copy()
        highest_reward_of_each_iteration.append(final_reward)
        # find the randomized order give us the min reward
        key_min = min(reward.keys(), key=(lambda k: reward[k]))
        min_randomized_order = population[key_min]
        del reward[key_min]

        # Crossover
        # Select the parent
        # Find two combinations of randomized order that give us highest reward
        max_key_a = max(reward.keys(), key=(lambda k: reward[k]))
        del reward[max_key_a]
        randomized_order_a = population[max_key_a]

        max_key_b = max(reward.keys(), key=(lambda k: reward[k]))
        del reward[max_key_b]
        randomized_order_b = population[max_key_b]

        # Crossover
        # Generate a random crossover
        cross_point = random.randrange(1,n_agents-1)
        first_part_order_a = randomized_order_a[:cross_point]
        second_part_order_a = randomized_order_a[cross_point:]

        first_part_order_b = randomized_order_b[:cross_point]
        second_part_order_b = randomized_order_b[cross_point:]

        # Create two children based on the genes of parent
        child_a = first_part_order_a + second_part_order_b
        child_b = first_part_order_b + second_part_order_a

        # Mutation
        mutation(child_a)
        mutation(child_b)

        # remove the two combinations in the population that have lower reward
        population.remove(min_randomized_order)
        population.remove(randomized_order_b)

        # add two new children to the population

        population.append(child_a)
        population.append(child_b)


    # Find the best schedule
    max_key_reward = max(final_reward.keys(), key=(lambda k: final_reward[k]))
    best_order = population[max_key_reward]

    print("--------------------------------Result-------------------------------------")
    print("Best Order: ", best_order)
    print("Reward: ",highest_reward_of_each_iteration)
    return best_order,action_plan[max_key_reward]

def mutation(child_order):
    """
    Apply mutation to other

    """
    duplicate_list = [[] for i in range(len(child_order))]
    # checking invalid gene
    for i in range(len(child_order)):
        duplicate_list[child_order[i]].append(i)

    index_of_invalid = []
    invalid_order = []
    for i in range(len(duplicate_list)):
        if len(duplicate_list[i]) != 1:
            invalid_order.append(i)
            if len(duplicate_list[i]) > 1:
                for index in duplicate_list[i]:
                    index_of_invalid.append(index)
    #if there is no invalid order
    if invalid_order == 0:
        # generate a random number
        rand_number = random.randint(1,100)
        # there is a 10% chance that mutation happen
        if rand_number < 10:
            # pick two agents randomly
            first_random_index = random.randint(0,len(child_order))
            second_random_index = random.randint (0,len(child_order))
            while first_random_index == second_random_index:
                second_random_index = random.randint(0, len(child_order))

            #swap the agents in the planning
            temp = child_order[first_random_index]
            child_order[first_random_index] = child_order[second_random_index]
            child_order[second_random_index] = temp
    else:
        # if there are invalid, we have to apply mutation
        for index in index_of_invalid:
            random_order = random.choice(invalid_order)
            child_order[index] = random_order
            invalid_order.remove(random_order)





