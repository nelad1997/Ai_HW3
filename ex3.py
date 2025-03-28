
ids = ["319000725", "207036211"]

import collections
import itertools
import time
RESET_REWARD = -2
DESTROY_HORCRUX_REWARD = 2
DEATH_EATER_CATCH_REWARD = -1
INIT_TIME_LIMIT = 200
BFS_LIMIT = 90


MOVE_ACTION = "move"
RESET_ACTION = "reset"
TERMINATE_ACTION = "terminate"
MOVE_UP = "move_up"
MOVE_DOWN = "move_down"
MOVE_LEFT = "move_left"
MOVE_RIGHT = "move_right"
DESTROY_ACTION = "destroy"
WAIT_ACTION = "wait"

class OptimalWizardAgent:
        def __init__(self, initial):
            """
            Initialize BFS, VI

            :param initial: The initial game-state dictionary (with map, wizards, horcrux, etc.).
            """
            self.start_time = time.perf_counter()
            self.turns_to_go = initial["turns_to_go"]
            self.grid = initial["map"]
            self.m = len(self.grid)  # number of rows
            self.n = len(self.grid[0])  # number of columns
            self.initial_state = initial
            self.wizards = initial["wizards"]
            self.wizard_names = sorted(self.wizards.keys())
            self.death_eaters = initial["death_eaters"]
            self.death_eater_names = sorted(self.death_eaters.keys())
            self.horcruxes = initial["horcrux"]
            self.h_names = sorted(self.horcruxes.keys())

            self.death_eater_paths = {}
            for de_nm in self.death_eater_names:
                self.death_eater_paths[de_nm] = self.death_eaters[de_nm]["path"]

            self.h_possible_loc_per_h = {}
            for h_nm in self.h_names:
                init_loc = self.horcruxes[h_nm]["location"]
                locs = set(self.horcruxes[h_nm]["possible_locations"]) | {init_loc}
                self.h_possible_loc_per_h[h_nm] = list(locs)

            self.passible_places = set()
            self.passible_loc()

            self.neighbors = {}
            self.neighbors_passible()

            # For BFS + VI
            self.values = [dict() for _ in range(self.turns_to_go + 1)]
            self.policy = [dict() for _ in range(self.turns_to_go + 1)]

            # Build the set of reachable states by BFS
            self.build_reachable_states_bfs(initial)

        def neighbors_passible(self):  # utility function
            directions = [(-1, 0), (0, 1), (0, -1), (1, 0)]
            for cell in self.passible_places:
                ne = []
                x, y = cell
                for dx, dy in directions:
                    new_x, new_y = x + dx, y + dy
                    if (new_x, new_y) in self.passible_places:
                        ne.append((new_x, new_y))
                self.neighbors[cell] = ne

        def passible_loc(self):  # utility function
            for i in range(self.m):
                for j in range(self.n):
                    if self.grid[i][j] != 'I':  # 'I' is Impassable
                        self.passible_places.add((i, j))

        def act(self, state):
            """
            Called each turn; must return a single action or a tuple-of-actions.
            We'll look up the best action from the policy if it exists.
            Otherwise, fallback to something simple (like wait).
            """
            time = state["turns_to_go"]

            state_reduced = self.reduce_state(state)


            return self.policy[time][state_reduced]

        def build_reachable_states_bfs(self, initial_state):
            """
            Builds the set of reachable states from the initial state using Breadth-First Search (BFS).
            The search continues until either the queue is empty or the time limit is reached.

            Returns:
                A set of reachable states, each represented as:
                ( (wizard_positions), (death_eater_indices), (horcrux_positions), turns_left )
            """
            initial_state_representation = self.reduce_state(initial_state)
            bfs_queue = collections.deque([initial_state_representation])
            reachable_states = set()
            reachable_states.add(initial_state_representation)

            while bfs_queue:
                current_state = bfs_queue.popleft()
                turns_left = current_state[-1]

                # If no turns are left, do not expand further
                if turns_left <= 0:
                    continue

                # Expand current state with all possible actions
                possible_actions = self.compute_possible_actions(current_state)
                for action in possible_actions:
                    transitions = self.compute_transition_probabilities(current_state, action)
                    for next_state, probability in transitions.items():
                        if probability > 0 and next_state not in reachable_states:
                            reachable_states.add(next_state)
                            bfs_queue.append(next_state)
            self.value_iteration(reachable_states)

        def value_iteration(self, reachable_states):
            """
            Performs the Value Iteration algorithm to compute the optimal policy for each reachable state.

            The process runs backward from the maximum number of turns to zero, calculating the best action
            for each state based on the expected cumulative reward.

            Parameters:
            reachable_states (set): A set of states discovered through BFS, where each state is represented
                                     as a tuple (wizard_locations, death_eater_indices, horcrux_locations, turns_left).
            """
            # Initialize terminal states with zero value
            for state_representation in reachable_states:
                if state_representation[-1] == 0:  # No turns left after last move
                    self.values[0][state_representation] = self.calculate_reward(None,
                                                                                 state_representation)  # for terminal state we cannot perform any action

            # Iterate backward from the last turn to the first
            for turns_remaining in range(1, self.turns_to_go + 1):
                # Filter states that have exactly 'turns_remaining' turns left

                states_at_current_turn = [state for state in reachable_states if state[-1] == turns_remaining]

                for current_state in states_at_current_turn:
                    possible_actions = self.compute_possible_actions(current_state)
                    if not possible_actions:
                        # If no actions available, assign zero value and force terminate
                        self.values[turns_remaining][current_state] = 0
                        self.policy[turns_remaining][current_state] = TERMINATE_ACTION
                        continue

                    # Track the best value and corresponding action
                    max_value = float('-inf')
                    best_action = None

                    for action in possible_actions:
                        transition_probabilities = self.compute_transition_probabilities(current_state, action)
                        expected_value = 0

                        # Immediate reward and the value of the current state
                        immediate_reward = self.calculate_reward(action, current_state)
                        expected_value += immediate_reward

                        if action != TERMINATE_ACTION:
                            for next_state, transition_probability in transition_probabilities.items():
                                if transition_probability <= 0:
                                    continue

                                # next_turns_remaining = next_state[-1]
                                future_value = self.values[turns_remaining - 1].get(next_state, 0.0)

                                # Bellman update equation
                                expected_value += transition_probability * future_value

                        # Update the best value and action if current action is better
                        if expected_value > max_value:
                            max_value = expected_value
                            best_action = action

                    # Store the optimal value and corresponding policy for the current state
                    self.values[turns_remaining][current_state] = max_value
                    self.policy[turns_remaining][current_state] = best_action


        def calculate_reward(self, actions, current_state):
            """
            Calculates the immediate reward for transitioning from the current state
            to the next state based on the given actions.

            Reward criteria:
            - RESET_ACTION results in a penalty of -2 points.
            - DESTROY_ACTION grants +2 points for each destroyed horcrux.
            - If a wizard encounters a death eater in the next state, -1 point is deducted for each encounter.

            :param actions: A list of actions taken by the wizards.
            :param current_state: The resulting state after performing the actions, represented as a tuple:
                               (wizard_positions, death_eater_indices, horcrux_positions, turns_left)
            :return: The calculated reward as an integer.
            """
            reward = 0
            (wizard_positions, death_eater_indices, horcrux_positions, turns_remaining) = current_state
            if actions != None and actions!=TERMINATE_ACTION:
                # Apply penalty for RESET action
                if actions == RESET_ACTION:
                    reward += RESET_REWARD

                else:
                    # Reward for destroying horcruxes
                    for action in actions:
                        if action[0] == DESTROY_ACTION:
                            reward += DESTROY_HORCRUX_REWARD

            # Penalty if a wizard is caught by a death eater in the next state
            for wizard_position in wizard_positions:
                for i, death_eater_name in enumerate(self.death_eater_names):
                    death_eater_path = self.death_eater_paths[death_eater_name]
                    death_eater_position = death_eater_path[death_eater_indices[i]]
                    if death_eater_position == wizard_position:
                        reward += DEATH_EATER_CATCH_REWARD

            return reward

        def reduce_state(self, state):
            # wizard positions
            wizard_locs = []
            for wizard_name in self.wizard_names:
                # stored as dict: wizards[name] -> {"location": (x,y)}
                wizard_locs.append(state["wizards"][wizard_name]["location"])
            wizard_locs = tuple(wizard_locs)

            # horcrux positions
            hocrux_locs = []
            for hocrux_name in self.h_names:
                hocrux_locs.append(state["horcrux"][hocrux_name]["location"])
            hocrux_locs = tuple(hocrux_locs)

            # death-eater indices
            death_eaters_idxs = []
            for de_name in self.death_eater_names:
                death_eaters_idxs.append(state["death_eaters"][de_name]["index"])
            death_eaters_idxs = tuple(death_eaters_idxs)

            time = state["turns_to_go"]

            return (wizard_locs, death_eaters_idxs, hocrux_locs, time)

        def compute_possible_actions(self, state):
            (wizard_locs, death_eaters_idxs, hocrux_locs, t) = state

            # always allow "reset" and "terminate"
            joint_actions = [RESET_ACTION, TERMINATE_ACTION]

            # build wizard actions
            wizard_actions_list = []
            for i, wizard_name in enumerate(self.wizard_names):
                wizard_pos = wizard_locs[i]
                act = []
                # moves
                if wizard_pos in self.neighbors:
                    for nb in self.neighbors[wizard_pos]:
                        act.append((MOVE_ACTION, wizard_name, nb))
                # wait
                act.append((WAIT_ACTION, wizard_name))
                # destroy if on horcrux
                for j, hocrux_name in enumerate(self.h_names):
                    if hocrux_locs[j] == wizard_pos:
                        act.append((DESTROY_ACTION, wizard_name, hocrux_name))
                wizard_actions_list.append(act)

            # cartesian product of wizard actions
            for combo in itertools.product(*wizard_actions_list):
                joint_actions.append(combo)

            return joint_actions

        def compute_transition_probabilities(self, state, actions):

            (wizard_locs, death_eaters_idxs, hocrux_locs, t) = state
            if t <= 0:
                return {state: 1.0}

            if actions == RESET_ACTION:
                init_state = self.reduce_state(self.get_initial_state())
                (iwizard, ideath, ihocrux, itime) = init_state
                # we came here from a reset action
                next_state = (iwizard, ideath, ihocrux, t - 1)
                return {next_state: 1.0}

            if actions == TERMINATE_ACTION:
                next_state = (wizard_locs, death_eaters_idxs, hocrux_locs, 0)
                return {next_state: 1.0}

            new_wiz_locs = list(wizard_locs)

            wizard_map = {nm: i for i, nm in enumerate(self.wizard_names)}
            for action in actions:
                if action[0] == MOVE_ACTION:
                    wiz_idx = wizard_map[action[1]]
                    new_wiz_locs[wiz_idx] = action[2]

            new_wiz_locs = tuple(new_wiz_locs)

            # death eater transitions

            all_death_eater_options = []
            for i, de_nm in enumerate(self.death_eater_names):
                all_death_eater_options.append(
                    self.get_death_eater_transition_probabilities(de_nm, death_eaters_idxs[i]))

            # horcrux transitions

            all_hocrux_options = []
            for i, h_nm in enumerate(self.h_names):
                all_hocrux_options.append(self.get_horcrux_transition_probabilities(h_nm, hocrux_locs[i]))

            next_states = {}

            for death_eater_combo in itertools.product(*all_death_eater_options):
                prob_death_eater = 1
                new_de_idxs = [None] * len(self.death_eater_names)
                for i, (ndi, pdi) in enumerate(death_eater_combo):
                    prob_death_eater *= pdi
                    new_de_idxs[i] = ndi
                if prob_death_eater <= 0:
                    continue
                new_de_idxs = tuple(new_de_idxs)

                for hocrux_combo in itertools.product(*all_hocrux_options):
                    prob_h = 1
                    new_h_locs = [None] * len(self.h_names)
                    for i, (hl, p_h) in enumerate(hocrux_combo):
                        prob_h *= p_h
                        new_h_locs[i] = hl
                    if prob_h <= 0:
                        continue

                    new_h_locs = tuple(new_h_locs)
                    p = prob_death_eater * prob_h

                    ns = (new_wiz_locs, new_de_idxs, new_h_locs, t - 1)
                    next_states[ns] = next_states.get(ns, 0) + p

            return next_states

        def get_horcrux_transition_probabilities(self, horcrux_name, current_location):
            """
            Calculates the transition probabilities for a given horcrux based on its probability of changing location.

            :param horcrux_name: The name of the horcrux.
            :param current_location: The current location of the horcrux.
            :return: A list of tuples representing possible new locations and their corresponding probabilities.
            """
            probability_change_location = self.horcruxes[horcrux_name]["prob_change_location"]
            possible_locations = self.h_possible_loc_per_h[horcrux_name]

            transition_probabilities = []
            stay_probability = 1 - probability_change_location
            move_probability = probability_change_location / len(possible_locations)

            for location in possible_locations:
                probability = move_probability
                if location == current_location:
                    probability += stay_probability  # Add stay probability if it's the current location
                if probability > 0:
                    transition_probabilities.append((location, probability))

            return transition_probabilities

        def get_death_eater_transition_probabilities(self, death_eater_name, current_index):
            """
            Calculates the transition probabilities for a death eater based on its current index along its path.

            :param death_eater_name: The name of the death eater.
            :param current_index: The current index of the death eater along its path.
            :return: A list of tuples representing possible new indices and their corresponding probabilities.
            """
            death_eater_path = self.death_eater_paths[death_eater_name]
            path_length = len(death_eater_path)

            if path_length == 1:
                return [(current_index, 1.0)]  # Only one position available, with probability 1

            if current_index == 0:
                return [(0, 0.5), (1, 0.5)]  # If at the start, can stay or move forward

            elif current_index == path_length - 1:
                return [(path_length - 1, 0.5), (path_length - 2, 0.5)]  # If at the end, can stay or move backward

            else:
                # In the middle, can move left, stay, or move right with equal probability
                return [
                    (current_index - 1, 1 / 3),
                    (current_index, 1 / 3),
                    (current_index + 1, 1 / 3)
                ]

        def get_initial_state(self):
            return self.initial_state




class WizardAgent():
    def __init__(self, initial):
        """
        Initialize BFS, VI

        :param initial: The initial game-state dictionary (with map, wizards, horcrux, etc.).
        """
        self.start_time = time.perf_counter()
        self.turns_to_go = initial["turns_to_go"]
        self.grid = initial["map"]
        self.m = len(self.grid) # number of rows
        self.n = len(self.grid[0]) # number of columns
        self.initial_state=initial
        self.wizards = initial["wizards"]
        self.wizard_names = sorted(self.wizards.keys())
        self.death_eaters = initial["death_eaters"]
        self.death_eater_names = sorted(self.death_eaters.keys())
        self.horcruxes = initial["horcrux"]
        self.h_names = sorted(self.horcruxes.keys())
        self.has_reset = False #for heuristic

        self.death_eater_paths = {}
        for de_nm in self.death_eater_names:
            self.death_eater_paths[de_nm] = self.death_eaters[de_nm]["path"]

        self.h_possible_loc_per_h = {}
        for h_nm in self.h_names:
            init_loc = self.horcruxes[h_nm]["location"]
            locs = set(self.horcruxes[h_nm]["possible_locations"]) | {init_loc}
            self.h_possible_loc_per_h[h_nm] = list(locs)


        self.passible_places = set()
        self.passible_loc()

        self.neighbors = {}
        self.neighbors_passible()

        # For BFS + VI
        self.values = [dict() for _ in range(self.turns_to_go + 1)]
        self.policy = [dict() for _ in range(self.turns_to_go + 1)]

        # Build the set of reachable states by BFS
        self.build_reachable_states_bfs(initial)

    def neighbors_passible(self):#utility function
        directions = [(-1, 0), (0, 1), (0, -1), (1, 0)]
        for cell in self.passible_places:
            ne = []
            x, y = cell
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                if (new_x, new_y) in self.passible_places:
                    ne.append((new_x, new_y))
            self.neighbors[cell] = ne

    def passible_loc(self):#utility function
        for i in range(self.m):
            for j in range(self.n):
                if self.grid[i][j] != 'I':  # 'I' is Impassable
                    self.passible_places.add((i, j))

    def act(self, state):
        """
        Called each turn; must return a single action or a tuple-of-actions.
        We'll look up the best action from the policy if it exists.
        Otherwise, fallback to something simple (like wait).
        """
        time = state["turns_to_go"]

        state_reduced = self.reduce_state(state)

        # If no policy use heuristic
        if state_reduced not in self.policy[time]:
            return self.heuristic(state)
        return self.policy[time][state_reduced]


    def build_reachable_states_bfs(self, initial_state):
        """
        Builds the set of reachable states from the initial state using Breadth-First Search (BFS).
        The search continues until either the queue is empty or the time limit is reached.

        Returns:
            A set of reachable states, each represented as:
            ( (wizard_positions), (death_eater_indices), (horcrux_positions), turns_left )
        """
        initial_state_representation = self.reduce_state(initial_state)
        bfs_queue = collections.deque([initial_state_representation])
        reachable_states = set()
        reachable_states.add(initial_state_representation)

        while bfs_queue:
            # Stop if time limit exceeded
            if time.perf_counter() - self.start_time > BFS_LIMIT:
                #print("we took too long in bfs")
                time_took_bfs = time.perf_counter() - self.start_time
                time_left_for_vi = INIT_TIME_LIMIT - time_took_bfs
                self.value_iteration(reachable_states, time_left_for_vi)
                return

            current_state = bfs_queue.popleft()
            turns_left = current_state[-1]

            # If no turns are left, do not expand further
            if turns_left <= 0:
                continue

            # Expand current state with all possible actions
            possible_actions = self.compute_possible_actions(current_state)
            for action in possible_actions:
                if time.perf_counter() - self.start_time > BFS_LIMIT:
                    #print("we took too long in bfs")
                    time_took_bfs = time.perf_counter() - self.start_time
                    time_left_for_vi = INIT_TIME_LIMIT - time_took_bfs
                    #print(f"time took to bfs {time_took_bfs} and time left for vi {time_left_for_vi}")
                    self.value_iteration(reachable_states,time_left_for_vi)
                    return

                bfs_Used_Time = time.perf_counter() - self.start_time
                time_left_for_bfs=BFS_LIMIT - bfs_Used_Time
                transitions = self.compute_transition_probabilities(current_state, action,time_left_for_bfs)
                if time.perf_counter() - self.start_time > BFS_LIMIT:
                    #print("we took too long in bfs")
                    time_took_bfs = time.perf_counter() - self.start_time
                    time_left_for_vi = INIT_TIME_LIMIT - time_took_bfs
                    #print(f"time took to bfs {time_took_bfs} and time left for vi {time_left_for_vi}")
                    self.value_iteration(reachable_states,time_left_for_vi)
                    return
                for next_state, probability in transitions.items():
                    if time.perf_counter() - self.start_time > BFS_LIMIT:
                        #print("we took too long in bfs")
                        time_took_bfs = time.perf_counter() - self.start_time
                        time_left_for_vi = INIT_TIME_LIMIT - time_took_bfs
                        #print(f"time took to bfs {time_took_bfs} and time left for vi {time_left_for_vi}")
                        self.value_iteration(reachable_states,time_left_for_vi)
                        return
                    if probability > 0 and next_state not in reachable_states:
                        reachable_states.add(next_state)
                        bfs_queue.append(next_state)

        time_took_bfs = time.perf_counter() - self.start_time
        time_left_for_vi = INIT_TIME_LIMIT - time_took_bfs
        #print(f"time took to bfs {time_took_bfs} and time left for vi {time_left_for_vi}")
        self.value_iteration(reachable_states,time_left_for_vi)


    def value_iteration(self, reachable_states,time_left_for_vi):
        """
        Performs the Value Iteration algorithm to compute the optimal policy for each reachable state.

        The process runs backward from the maximum number of turns to zero, calculating the best action
        for each state based on the expected cumulative reward.

        Parameters:
        reachable_states (set): A set of states discovered through BFS, where each state is represented
                                 as a tuple (wizard_locations, death_eater_indices, horcrux_locations, turns_left).
        """
        start_vi_time = time.perf_counter()
        if time.perf_counter() - start_vi_time > time_left_for_vi:
                #print("we didnt start vi")
                return
        # Initialize terminal states with zero value
        for state_representation in reachable_states:
            if time.perf_counter() - start_vi_time > time_left_for_vi:
                #print(f"we got to turn {0} from {self.turns_to_go} turns remaining ")
                return
            if state_representation[-1] == 0:  # No turns left after last move
                self.values[0][state_representation] = self.calculate_reward(None , state_representation)#for terminal state we cannot perform any action

        # Iterate backward from the last turn to the first
        for turns_remaining in range(1, self.turns_to_go + 1):
            # Filter states that have exactly 'turns_remaining' turns left
            if time.perf_counter() - start_vi_time > time_left_for_vi:
                #print(f"we got to turn {turns_remaining} from {self.turns_to_go} turns remaining ")
                return

            states_at_current_turn = [state for state in reachable_states if state[-1] == turns_remaining]

            if time.perf_counter() - start_vi_time > time_left_for_vi:
                #print(f"we got to turn {turns_remaining} from {self.turns_to_go} turns remaining ")
                return

            for current_state in states_at_current_turn:
                # Terminate if time limit is exceeded
                if time.perf_counter() - start_vi_time > time_left_for_vi:
                    #print(f"we got to turn {turns_remaining} from {self.turns_to_go} turns remaining ")
                    return

                possible_actions = self.compute_possible_actions(current_state)
                if not possible_actions:
                    # If no actions available, assign zero value and force terminate
                    self.values[turns_remaining][current_state] = 0
                    self.policy[turns_remaining][current_state] = TERMINATE_ACTION
                    continue

                # Track the best value and corresponding action
                max_value = float('-inf')
                best_action = None

                for action in possible_actions:
                    if time.perf_counter() - start_vi_time > time_left_for_vi:
                        #print(f"we got to turn {turns_remaining} from {self.turns_to_go} turns remaining ")
                        return
                    vi_used_time = time.perf_counter() - start_vi_time
                    time_left_for_init= INIT_TIME_LIMIT - vi_used_time
                    transition_probabilities = self.compute_transition_probabilities(current_state, action,time_left_for_init)
                    expected_value = 0

                    if time.perf_counter() - start_vi_time > time_left_for_vi:
                        #print(f"we got to turn {turns_remaining} from {self.turns_to_go} turns remaining ")
                        return

                    # Immediate reward and the value of the current state
                    immediate_reward = self.calculate_reward(action, current_state)
                    expected_value += immediate_reward
                    if action!=TERMINATE_ACTION:
                        for next_state, transition_probability in transition_probabilities.items():

                            if time.perf_counter() - start_vi_time > time_left_for_vi:
                                #print(f"we got to turn {turns_remaining} from {self.turns_to_go} turns remaining ")
                                return
                            if transition_probability <= 0:
                                continue

                            # next_turns_remaining = next_state[-1]
                            future_value = self.values[turns_remaining - 1].get(next_state, 0.0)

                            # Bellman update equation
                            expected_value += transition_probability * future_value

                    # Update the best value and action if current action is better
                    if expected_value > max_value:
                        max_value = expected_value
                        best_action = action

                # Store the optimal value and corresponding policy for the current state
                self.values[turns_remaining][current_state] = max_value
                self.policy[turns_remaining][current_state] = best_action

        time_took_vi = time.perf_counter() - start_vi_time
        #print(f"time took to vi {time_took_vi} and time left for init {INIT_TIME_LIMIT-time_took_vi}")

    def calculate_reward(self, actions, current_state):
        """
        Calculates the immediate reward for transitioning from the current state
        to the next state based on the given actions.

        Reward criteria:
        - RESET_ACTION results in a penalty of -2 points.
        - DESTROY_ACTION grants +2 points for each destroyed horcrux.
        - If a wizard encounters a death eater in the next state, -1 point is deducted for each encounter.

        :param actions: A list of actions taken by the wizards.
        :param current_state: The resulting state after performing the actions, represented as a tuple:
                           (wizard_positions, death_eater_indices, horcrux_positions, turns_left)
        :return: The calculated reward as an integer.
        """
        reward = 0
        (wizard_positions, death_eater_indices, horcrux_positions, turns_remaining) = current_state
        if actions != None and actions!=TERMINATE_ACTION:
            # Apply penalty for RESET action
            if actions == RESET_ACTION:
                reward += RESET_REWARD
            else:
                # Reward for destroying horcruxes
                for action in actions:
                    if action[0] == DESTROY_ACTION:
                        reward += DESTROY_HORCRUX_REWARD

        # Penalty if a wizard is caught by a death eater in the next state
        for wizard_position in wizard_positions:
            for i, death_eater_name in enumerate(self.death_eater_names):
                death_eater_path = self.death_eater_paths[death_eater_name]
                death_eater_position = death_eater_path[death_eater_indices[i]]
                if death_eater_position == wizard_position:
                    reward += DEATH_EATER_CATCH_REWARD

        return reward

    def reduce_state(self, state):
        # wizard positions
        wizard_locs = []
        for wizard_name in self.wizard_names:
            # stored as dict: wizards[name] -> {"location": (x,y)}
            wizard_locs.append(state["wizards"][wizard_name]["location"])
        wizard_locs = tuple(wizard_locs)


        # horcrux positions
        hocrux_locs = []
        for hocrux_name in self.h_names:
            hocrux_locs.append(state["horcrux"][hocrux_name]["location"])
        hocrux_locs = tuple(hocrux_locs)

        # death-eater indices
        death_eaters_idxs = []
        for de_name in self.death_eater_names:
            death_eaters_idxs.append(state["death_eaters"][de_name]["index"])
        death_eaters_idxs = tuple(death_eaters_idxs)

        time = state["turns_to_go"]

        return (wizard_locs, death_eaters_idxs, hocrux_locs, time)

    def heuristic(self, state):

        t = state["turns_to_go"]
        if t <= 0:
            return TERMINATE_ACTION

        jactions = []

        # Gather current death eater positions (to estimate q(a))
        death_eater_positions = set()
        for de_nm in self.death_eater_names:
            de_idx = state["death_eaters"][de_nm]["index"]
            de_pos = self.death_eater_paths[de_nm][de_idx]
            death_eater_positions.add(de_pos)

        # Check if a reset has already been performed (if not, default to False)
        reset_flag = self.has_reset

        # Compute a dynamic alpha: here we set alpha as 1 divided by the number
        # of all distinct possible horcrux locations (real + possible)
        horcrux_locs_all = set()
        for h_name in self.h_names:
            h_real = state["horcrux"][h_name]["location"]
            horcrux_locs_all.add(h_real)
            horcrux_locs_all.update(self.h_possible_loc_per_h[h_name])
        possible_count = len(horcrux_locs_all)
        prob_h = 1.0 / possible_count if possible_count > 0 else 1.0

        # Process each wizard individually
        for wizard_name in self.wizard_names:
            w_loc = state["wizards"][wizard_name]["location"]


            destroy_actions = []
            for h_name in self.h_names:
                real_location = state["horcrux"][h_name]["location"]
                if real_location == w_loc:
                    destroy_actions.append((DESTROY_ACTION, wizard_name, h_name))
            if destroy_actions:
                jactions.append(destroy_actions[0])
                continue


            best_hor_score = float('-inf')
            best_hor_loc = None

            for h_name in self.h_names:
                h_real = state["horcrux"][h_name]["location"]
                dist = abs(w_loc[0] - h_real[0]) + abs(w_loc[1] - h_real[1])  # Manhattan distance
                p_d = prob_h ** dist
                if p_d > best_hor_score:
                    best_hor_score = p_d
                    best_hor_loc = h_real

            if best_hor_loc is None:
                jactions.append((WAIT_ACTION, wizard_name))
                continue

            best_score = float('-inf')
            best_action = (WAIT_ACTION, wizard_name)
            possible_actions = []
            for nb in self.neighbors[w_loc]:
                possible_actions.append((MOVE_ACTION, wizard_name, nb))
            possible_actions.append((WAIT_ACTION, wizard_name))

            for act in possible_actions:
                is_reset = 1 if (act == RESET_ACTION) else 0

                # Compute q(a): if the action is a MOVE and the target cell is occupied by a death eater
                if  act[0] == MOVE_ACTION:
                    _, _, target_cell = act
                    q_a = 1 if (target_cell in death_eater_positions) else 0
                    new_loc = target_cell
                elif  act[0] in [WAIT_ACTION, DESTROY_ACTION]:
                    new_loc = w_loc
                    q_a = 0
                elif act == RESET_ACTION:
                    new_loc = w_loc  # or self.initial["wizards"][w_nm]["location"]
                    q_a = 0
                else:
                    new_loc = w_loc
                    q_a = 0

                # Compute the distance from the new location to the best horcrux location
                dist2 = abs(new_loc[0] - best_hor_loc[0]) + abs(new_loc[1] - best_hor_loc[1])
                p_d2 = prob_h ** dist2

                partial_score = p_d2 * 2  # Reward for destroying if reached
                score = partial_score - 2 * is_reset - q_a
                if reset_flag:
                    score -= 2

                if score > best_score:
                    best_score = score
                    best_action = act

            if best_action == RESET_ACTION:
                self.has_reset = True
                return RESET_ACTION
            else:
                jactions.append(best_action)

        # Ensure we return exactly one atomic command per wizard
        return tuple(jactions)

    def compute_possible_actions(self, state):
        (wizard_locs, death_eaters_idxs, hocrux_locs, t) = state

        # always allow "reset" and "terminate"
        joint_actions = [RESET_ACTION, TERMINATE_ACTION]

        # build wizard actions
        wizard_actions_list = []
        for i, wizard_name in enumerate(self.wizard_names):
            wizard_pos = wizard_locs[i]
            act = []
            # moves
            if wizard_pos in self.neighbors:
                for nb in self.neighbors[wizard_pos]:
                    act.append((MOVE_ACTION, wizard_name, nb))
            # wait
            act.append((WAIT_ACTION, wizard_name))
            # destroy if on horcrux
            for j, hocrux_name in enumerate(self.h_names):
                if hocrux_locs[j] == wizard_pos:
                    act.append((DESTROY_ACTION, wizard_name, hocrux_name))
            wizard_actions_list.append(act)

        # cartesian product of wizard actions
        for combo in itertools.product(*wizard_actions_list):
            joint_actions.append(combo)

        return joint_actions

    def compute_transition_probabilities(self, state, actions,time_left):
        start_prob_time=time.perf_counter()
        (wizard_locs, death_eaters_idxs, hocrux_locs, t) = state
        if t <= 0:
            return {state: 1.0}

        if actions == RESET_ACTION:
            init_state = self.reduce_state(self.get_initial_state())
            (iwizard, ideath_, ihocrux, itime) = init_state
            #we came here from a reset action
            next_state = (iwizard, ideath_, ihocrux, t - 1)
            return {next_state: 1.0}

        if actions == TERMINATE_ACTION:
            next_state = (wizard_locs, death_eaters_idxs, hocrux_locs, 0)
            return {next_state: 1.0}


        new_wiz_locs = list(wizard_locs)

        wizard_map = {nm: i for i, nm in enumerate(self.wizard_names)}
        for action in actions:
            if time.perf_counter()-start_prob_time > time_left:
                return {state: 1.0}
            if action[0] == MOVE_ACTION:
                wiz_idx = wizard_map[action[1]]
                new_wiz_locs[wiz_idx] = action[2]

        new_wiz_locs = tuple(new_wiz_locs)

        # death eater transitions

        all_death_eater_options = []
        for i, de_nm in enumerate(self.death_eater_names):
            if time.perf_counter()-start_prob_time > time_left:
                return {state: 1.0}
            all_death_eater_options.append(self.get_death_eater_transition_probabilities(de_nm, death_eaters_idxs[i]))

        # horcrux transitions

        all_hocrux_options = []
        for i, h_nm in enumerate(self.h_names):
            if time.perf_counter()-start_prob_time > time_left:
                return {state: 1.0}
            all_hocrux_options.append(self.get_horcrux_transition_probabilities(h_nm, hocrux_locs[i]))

        next_states = {}

        for death_eater_combo in itertools.product(*all_death_eater_options):
            if time.perf_counter()-start_prob_time > time_left:
                return {state: 1.0}
            prob_death_eater = 1
            new_de_idxs = [None]*len(self.death_eater_names)
            for i, (ndi, pdi) in enumerate(death_eater_combo):
                prob_death_eater *= pdi
                new_de_idxs[i] = ndi
            if prob_death_eater <= 0:
                continue
            new_de_idxs = tuple(new_de_idxs)

            for hocrux_combo in itertools.product(*all_hocrux_options):
                if time.perf_counter() - start_prob_time > time_left:
                    return {state: 1.0}
                prob_h = 1
                new_h_locs = [None]*len(self.h_names)
                for i, (hl, p_h) in enumerate(hocrux_combo):
                    prob_h *= p_h
                    new_h_locs[i] = hl
                if prob_h <= 0:
                    continue

                new_h_locs = tuple(new_h_locs)
                p = prob_death_eater * prob_h

                ns = (new_wiz_locs, new_de_idxs, new_h_locs, t-1)
                next_states[ns] = next_states.get(ns, 0) + p

        return next_states

    def get_horcrux_transition_probabilities(self, horcrux_name, current_location):
        """
        Calculates the transition probabilities for a given horcrux based on its probability of changing location.

        :param horcrux_name: The name of the horcrux.
        :param current_location: The current location of the horcrux.
        :return: A list of tuples representing possible new locations and their corresponding probabilities.
        """
        probability_change_location = self.horcruxes[horcrux_name]["prob_change_location"]
        possible_locations = self.h_possible_loc_per_h[horcrux_name]

        transition_probabilities = []
        stay_probability = 1 - probability_change_location
        move_probability = probability_change_location / len(possible_locations)

        for location in possible_locations:
            probability = move_probability
            if location == current_location:
                probability += stay_probability  # Add stay probability if it's the current location
            if probability > 0:
                transition_probabilities.append((location, probability))

        return transition_probabilities

    def get_death_eater_transition_probabilities(self, death_eater_name, current_index):
        """
        Calculates the transition probabilities for a death eater based on its current index along its path.

        :param death_eater_name: The name of the death eater.
        :param current_index: The current index of the death eater along its path.
        :return: A list of tuples representing possible new indices and their corresponding probabilities.
        """
        death_eater_path = self.death_eater_paths[death_eater_name]
        path_length = len(death_eater_path)

        if path_length == 1:
            return [(current_index, 1.0)]  # Only one position available, with probability 1

        if current_index == 0:
            return [(0, 0.5), (1, 0.5)]  # If at the start, can stay or move forward

        elif current_index == path_length - 1:
            return [(path_length - 1, 0.5), (path_length - 2, 0.5)]  # If at the end, can stay or move backward

        else:
            # In the middle, can move left, stay, or move right with equal probability
            return [
                (current_index - 1, 1 / 3),
                (current_index, 1 / 3),
                (current_index + 1, 1 / 3)
            ]


    def get_initial_state(self):
        return self.initial_state