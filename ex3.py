
ids = ["319000725", "207036211"]

import collections
import itertools
import time
RESET_REWARD = -2
DESTROY_HORCRUX_REWARD = 2
DEATH_EATER_CATCH_REWARD = -1
INIT_TIME_LIMIT = 140
BFS_LIMIT = 80


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
        Constructor for the OptimalWizardAgent class.
        Initializes the game state, precomputes the reduced states, and prepares the transition matrices.
        Ensures time constraints are respected.
        """
        self.start = time.perf_counter()  # Start time for time tracking
        self.turns_to_go = initial["turns_to_go"]
        self.grid = initial["map"]
        self.m = len(initial['map'])
        self.n = len(initial['map'][0])
        self.wizards = initial["wizards"]
        self.list_wizards = list(self.wizards.keys())  # Static wizard locations
        self.horcruxes = initial["horcrux"]
        self.list_horcruxes = list(self.horcruxes.keys())
        self.death_eaters = initial["death_eaters"]
        self.list_death_eaters = list(self.death_eaters.keys())
        self.actions = ["move_up", "move_down", "move_left", "move_right", "destroy", "wait"]
        self.values_regular={} #for the regular problem
        self.policy_regular={} #for the regular problem
        self.joint_actions = self.generate_joint_actions()
        self.states = self.generate_all_states()
        self.reduced_states = {self.reduce_state(state): state for state in self.states}
        self.initial_state = self.reduce_state(initial)#intial reduced state
        self.transition_matrices = self.build_transition_matrices()
        self.actions_cache = self.cache_all_possible_actions()
        self.value_iteration()

    def cache_all_possible_actions(self):
        """
        Precomputes all possible actions for every state in reduced_states.
        """
        actions_cache = {}
        for reduced_state, state in self.reduced_states.items():
            actions_cache[reduced_state] = list(self.all_possible_actions(state))
        return actions_cache

    def generate_joint_actions(self):
        actions_per_player = [self.actions for _ in range(len(self.wizards))]
        return tuple(itertools.product(*actions_per_player))

    def generate_all_states(self):
        """
        יוצר את כל המצבים האפשריים במשחק, תוך התחשבות במגבלות כגון מיקום קוסמים וחסמים במפה.
        """
        # סינון מיקומים אפשריים לקוסמים (רק מיקומי "P" במפה)
        valid_positions = self.grid_positions()  # כל המיקומים האפשריים ברשת (ללא "I")

        # יצירת כל הקומבינציות האפשריות של מיקומי הקוסמים
        wizard_combinations = itertools.product(valid_positions, repeat=len(self.wizards))
        wizard_states = [
            {name: pos for name, pos in zip(self.list_wizards, combination)}
            for combination in wizard_combinations
        ]

        # יצירת כל הקומבינציות האפשריות של אוכלי המוות (על בסיס המסלולים שלהם)
        death_eater_states = itertools.product(*[
            range(len(data["path"])) for data in self.death_eaters.values()
        ])
        death_eater_states_named = [
            {name: index for name, index in zip(self.list_death_eaters, combination)}
            for combination in death_eater_states
        ]

        # יצירת כל הקומבינציות האפשריות של מיקומי ה-Horcruxes
        horcrux_combinations = itertools.product(*[
            data["possible_locations"] for data in self.horcruxes.values()
        ])
        horcrux_states = [
            {name: loc for name, loc in zip(self.list_horcruxes, combination)}
            for combination in horcrux_combinations
        ]

        # יצירת כל המצבים האפשריים עם סינון מוקדם של מיקומים לא תקפים
        return [
            {
                "wizards": wizards,
                "death_eaters": death_eaters,
                "horcrux": horcruxes
            }
            for wizards in wizard_states
            for death_eaters in death_eater_states_named
            for horcruxes in horcrux_states
        ]

    def grid_positions(self):
        positions = []
        for x, row in enumerate(self.grid):
            for y, cell in enumerate(row):
                if cell != "I":  # I = Impassable
                    positions.append((x, y))
        return positions

    def build_transition_matrices(self):
        matrices = {}
        for joint_action in self.joint_actions:
            matrices[joint_action] = self.build_transition_matrix_for_action(joint_action)
        matrices["reset"] = {}

        for reduced_state in self.reduced_states.keys():
            # המצב המקורי (המצב ההתחלתי)
            initial_state = self.initial_state

            # הוספת מעבר מהמקרה הנוכחי לסטייט ההתחלתי
            matrices["reset"][reduced_state] = {
                initial_state: 1.0  # הסתברות 1
            }
        return matrices

    def build_transition_matrix_for_action(self, joint_action):
        transition_matrix = {}
        for reduced_state, original_state in self.reduced_states.items():
            transition_matrix[reduced_state] = {}
            for reduced_next_state, next_original_state in self.reduced_states.items():
                probability = self.calculate_transition_probability(original_state, next_original_state, joint_action)
                transition_matrix[reduced_state][reduced_next_state] = probability
        return transition_matrix

    def calculate_transition_probability(self, state_from, state_to, joint_action):
        total_probability = 1

        wizards_from = state_from["wizards"]
        wizards_to = state_to["wizards"]
        death_eaters_from = state_from["death_eaters"]
        death_eaters_to = state_to["death_eaters"]
        horcruxes_from = state_from["horcrux"]
        horcruxes_to = state_to["horcrux"]

        for wizard_name, action in zip(self.list_wizards, joint_action):#the wizards have a static locations in the list!
            pos_from = wizards_from[wizard_name]
            pos_to = wizards_to[wizard_name]
            probability = self.calculate_wizard_action_probability(pos_from, pos_to, action,wizard_name,horcruxes_from)
            if probability == 0:
                return 0
            total_probability *= probability

        for death_eater, index_from in death_eaters_from.items():
            index_to = death_eaters_to[death_eater]
            probability = self.calculate_death_eater_transition_probability(index_from, index_to, death_eater)
            if probability == 0:
                return 0
            total_probability *= probability

        for horcrux, position_from in horcruxes_from.items():
            position_to = horcruxes_to[horcrux]
            probability = self.calculate_horcrux_transition_probability(position_from, position_to, horcrux)
            if probability == 0:
                return 0
            total_probability *= probability

        return total_probability

    def calculate_wizard_action_probability(self, pos_from, pos_to, action,wizard_name,horcruxes):
        possible_actions_for_wizard=self.possible_actions(wizard_name,pos_from,horcruxes,self.grid,self.n,self.m)
        for action_p in possible_actions_for_wizard:#action_p is a correct action
            if action==action_p[0]:
                if action=="wait":
                    return pos_to==pos_from
                elif action=="destroy":
                    return pos_from==pos_to and horcruxes[action_p[2]]==pos_from
                else:
                    return action_p[2]==pos_to
        return 0


    def calculate_death_eater_transition_probability(self, index_from, index_to, death_eater):
        path = self.death_eaters[death_eater]["path"]
        possible_indices = [index_from]
        if index_from > 0:
            possible_indices.append(index_from - 1)
        if index_from < len(path) - 1:
            possible_indices.append(index_from + 1)
        return 1 / len(possible_indices) if index_to in possible_indices else 0

    def calculate_horcrux_transition_probability(self, position_from, position_to, horcrux):
        horcrux_data = self.horcruxes[horcrux]
        possible_locations = horcrux_data["possible_locations"]
        prob_change = horcrux_data["prob_change_location"]

        if position_to == position_from:
            return 1 - prob_change + (prob_change / len(possible_locations))
        elif position_to in possible_locations:
            return prob_change / len(possible_locations)
        return 0


    def reduce_state(self, state):
        """
        ממיר מצב לפורמט מצומצם שמתחשב במבנה הנתון של הבעיה.
        """
        # מיקום הקוסמים
        wizards = tuple(
            (name, tuple(data["location"]) if isinstance(data, dict) else tuple(data))
            for name, data in state["wizards"].items()
        )

        # מיקום ההורקרוקסים
        horcruxes = tuple(
            (name, tuple(data["location"]) if isinstance(data, dict) else tuple(data))
            for name, data in state["horcrux"].items()
        )

        # מיקום אוכלי המוות
        death_eaters = tuple(
            (name, data["index"] if isinstance(data, dict) else data)
            for name, data in state["death_eaters"].items()
        )

        return (wizards, horcruxes, death_eaters)

    def act(self, state):
        key_state = self.reduce_state(state)
        t=self.turns_to_go
        chosen_action = self.policy.get((key_state, t))
        self.turns_to_go-=1
        return chosen_action

    def find_destroyable_horcrux(self, state, wizard):
        wizards = state["wizards"]
        horcrux = state["horcrux"]

        wizard_position = wizards[wizard]
        for horcrux, position in horcrux.items():
            if wizard_position == position:
                return horcrux
        return None

    def get_target_position(self, current_position, action):
        """
        מחשבת את המיקום הבא של הקוסם בהתאם לפעולה.
        """
        x, y = current_position
        moves = {
            "move_up": (x - 1, y),
            "move_down": (x + 1, y),
            "move_left": (x, y - 1),
            "move_right": (x, y + 1),
        }
        target_position = moves.get(action, current_position)  # פעולה שלא מוגדרת תגרום להישארות במקום
        # בדיקה אם המיקום תקף ברשת (grid)
        if target_position in self.grid_positions():
            return target_position
        return current_position

    def possible_actions(self, wizard_name, wizard_pos, horcruxes, gmap, n, m):
        """
        returns a list of possible actions for wizard
        horcruxes is a dictionary of the horcruxes in the game with all their details
        wizard_name is the name of the wizard
        wizard_pos is the position of the wizard
        """
        actions = []
        current_place = wizard_pos
        wName = wizard_name
        numCol = n - 1
        numRow = m - 1
        x = current_place[0]
        y = current_place[1]

        actions.append(("wait", wName))  # אפשרות לחכות

        # בדיקה של כל הפעולות האפשריות
        if x + 1 <= numRow and gmap[x + 1][y] != 'I':  # תזוזה למטה
            actions.append(("move_down", wName, (x + 1, y)))
        if 0 <= x - 1 and gmap[x - 1][y] != 'I':  # תזוזה למעלה
            actions.append(("move_up", wName, (x - 1, y)))
        if 0 <= y - 1 and gmap[x][y - 1] != 'I':  # תזוזה שמאלה
            actions.append(("move_left", wName, (x, y - 1)))
        if y + 1 <= numCol and gmap[x][y + 1] != 'I':  # תזוזה ימינה
            actions.append(("move_right", wName, (x, y + 1)))

        # בדיקה אם אפשר להרוס הוקרוקס
        for horcrux_name ,horcrux_pos in horcruxes.items():
            if (x, y) == horcrux_pos:
                actions.append(("destroy", wName, horcrux_name))
        return actions

    def all_possible_actions(self, state):
        all_wizards_actions = []
        for wizard, wizard_pos in state['wizards'].items():  # לכל קוסם, חשב את כל הפעולות האפשריות
            possible = self.possible_actions(wizard, wizard_pos, state['horcrux'], self.grid, self.n, self.m)
            all_wizards_actions.append(possible)
        # ייצור כל הקומבינציות האפשריות של הפעולות
        all_combinations = itertools.product(*all_wizards_actions)
        return all_combinations


    def Reward(self, state, actions,t): #It is also depends on the action we want to make from state s abd t
        """
        returns the reward for the given actions and the current state
        actions is a vector of action for each one of the wizards from the current state
        state in its original form and not the reduced form of the current state
        t is the number of the turns we still have
        """
        reward = 0
        for wizard, wizard_pos in state["wizards"].items():#depends on the current state we are in
            for death_eater_name, index_death_eater in state["death_eaters"].items():#data_death_eater includes the death eater path and the current index in the path
                death_eater_pos = self.death_eaters[death_eater_name]["path"][index_death_eater]  # המיקום במסלול לפי האינדקס
                if wizard_pos == death_eater_pos:
                    reward += DEATH_EATER_CATCH_REWARD
        if t!=0 :
            if actions=='reset':
                reward +=RESET_REWARD
                return reward

            for action in actions:
                if action[0]=='destroy':
                    reward += DESTROY_HORCRUX_REWARD
        return reward

    def normalize_action(self,action):
        # אם הפעולה מתחילה ב-"move_", היא תומר ל-"move" בלבד
        if action.startswith("move_"):
            return "move"
        return action  # פעולות אחרות מוחזרות כמו שהן

    def process_joint_actions(self,joint_actions):
        """
        מטפלת בטאפל של טאפלים, מחליפה את שם הפעולה (הממוקם במקום ה-0 בכל טאפל)
        למצב המעובד (למשל, "move_up" -> "move").

        Args:
            joint_actions (tuple of tuples): טאפלים של פעולות, למשל:
                (("move_up", "Harry"), ("destroy", "Ron"), ("wait", "Hermione"))

        Returns:
            tuple of tuples: הטאפלים לאחר עיבוד הפעולה:
                (("move", "Harry"), ("destroy", "Ron"), ("wait", "Hermione"))
        """

        # יצירת טאפלים חדשים לאחר עיבוד
        if joint_actions=="reset":
            return "reset"
        processed_actions = tuple(
            (self.normalize_action(action[0]), *action[1:])  # שינוי הפעולה והחזרת שאר המידע
            for action in joint_actions
        )
        return processed_actions

    def value_iteration(self):
        """
        מבצע איטרציה על הערכים ומחשב את המדיניות הטובה ביותר.
        """
        self.values = {}
        self.policy = {}

        # אתחול קאש של כל הפעולות האפשריות
        actions_cache = self.cache_all_possible_actions()

        # אתחול ערכים עבור t=0
        for reduced_state, state in self.reduced_states.items():
            self.values[(reduced_state, 0)] = self.Reward(state, None, 0)

        # איטרציה על זמן t
        for t in range(1, self.turns_to_go + 1):
            for reduced_state, state in self.reduced_states.items():
                max_value = float("-inf")
                best_action = None

                # קבלת פעולות אפשריות מהקאש
                all_possible_actions = list(self.actions_cache[reduced_state])
                all_possible_actions.append("reset")
                all_possible_actions.append("terminate")

                for actions in all_possible_actions:

                    current_state_value = self.Reward(state, actions, t)
                    if actions != "terminate":
                        if actions == "reset":
                            actions_key = "reset"
                        else:
                            actions_key = tuple(action[0] for action in actions)
                        action_matrix = self.transition_matrices[actions_key]
                        transition_probabilities = action_matrix[reduced_state]

                        for reduced_state_next, prob in transition_probabilities.items():
                            current_state_value += prob * self.values[(reduced_state_next, t - 1)]

                    if current_state_value > max_value:
                        max_value = current_state_value
                        best_action = actions

                self.values[(reduced_state, t)] = max_value
                self.policy[(reduced_state, t)] = self.process_joint_actions(best_action)




class WizardAgent():
    def __init__(self, initial):
        """
        Initialize BFS, VI

        :param initial: The initial game-state dictionary (with map, wizards, horcrux, etc.).
        """
        self.start_time = time.perf_counter()
        self.turns_to_go = initial["turns_to_go"]
        self.grid = initial["map"]
        self.m = len(self.grid)         # number of rows
        self.n = len(self.grid[0])      # number of columns
        self.initial_state=initial
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
        self.values = [dict() for i in range(self.turns_to_go + 1)]
        self.policy = [dict() for i in range(self.turns_to_go + 1)]

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

        # If no policy entry use heuristic
        if time >= len(self.policy) or state_reduced not in self.policy[time]:
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

                transitions = self.compute_transition_probabilities(current_state, action)
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
                self.values[0][state_representation] = 0

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
                    transition_probabilities = self.compute_transition_probabilities(current_state, action)
                    expected_value = 0

                    if time.perf_counter() - start_vi_time > time_left_for_vi:
                        #print(f"we got to turn {turns_remaining} from {self.turns_to_go} turns remaining ")
                        return

                    for next_state, transition_probability in transition_probabilities.items():
                        if time.perf_counter() - start_vi_time > time_left_for_vi:
                            #print(f"we got to turn {turns_remaining} from {self.turns_to_go} turns remaining ")
                            return
                        if transition_probability <= 0:
                            continue

                        # Immediate reward and the value of the next state
                        immediate_reward = self.calculate_reward(action, next_state)
                        next_turns_remaining = next_state[-1]
                        future_value = self.values[next_turns_remaining].get(next_state, 0.0)

                        # Bellman update equation
                        expected_value += transition_probability * (immediate_reward + future_value)

                    # Update the best value and action if current action is better
                    if expected_value > max_value:
                        max_value = expected_value
                        best_action = action

                # Store the optimal value and corresponding policy for the current state
                self.values[turns_remaining][current_state] = max_value
                self.policy[turns_remaining][current_state] = best_action

        #time_took_vi = time.perf_counter() - start_vi_time
        #print(f"time took to vi {time_took_vi} and time left for init {INIT_TIME_LIMIT-time_took_vi}")


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
        reset_flag = getattr(self, "has_reset", False)

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
                jactions.append(RESET_ACTION)
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

    def compute_transition_probabilities(self, state, actions):

        (wizard_locs, death_eaters_idxs, h0crux_locs, t) = state
        if t <= 0:
            return {state: 1.0}

        if actions == RESET_ACTION:
            init_state = self.reduce_state(self.get_initial_state())
            (iwizard, ideath_, ihocrux, itime) = init_state
            #we came here from a reset action
            next_state = (iwizard, ideath_, ihocrux, t - 1)
            return {next_state: 1.0}

        if actions == TERMINATE_ACTION:
            next_state = (wizard_locs, death_eaters_idxs, h0crux_locs, 0)
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
            all_death_eater_options.append(self.get_deaeth_eater_p(de_nm, death_eaters_idxs[i]))

        # horcrux transitions

        all_hocrux_options = []
        for i, h_nm in enumerate(self.h_names):
            all_hocrux_options.append(self.get_hocrux_p(h_nm, h0crux_locs[i]))

        next_states = {}

        for death_eater_combo in itertools.product(*all_death_eater_options):
            prob_death_eater = 1
            new_de_idxs = [None]*len(self.death_eater_names)
            for i, (ndi, pdi) in enumerate(death_eater_combo):
                prob_death_eater *= pdi
                new_de_idxs[i] = ndi
            if prob_death_eater <= 0:
                continue
            new_de_idxs = tuple(new_de_idxs)

            for hocrux_combo in itertools.product(*all_hocrux_options):
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

    def get_hocrux_p(self,hocrux_name, current_loc):
        pc = self.horcruxes[hocrux_name]["prob_change_location"]
        locs = self.h_possible_loc_per_h[hocrux_name]
        # stay with prob (1 - pc), otherwise uniform among locs
        # so total for move among locs is pc / len(locs)
        res = []
        stay_prob = 1 - pc
        move_p = pc / len(locs)
        for loc_ in locs:
            p = move_p
            if loc_ == current_loc:
                p += stay_prob
            if p > 0:
                res.append((loc_, p))
        return res

    def get_deaeth_eater_p(self,deathEater_Nmae, idx):
        path = self.death_eater_paths[deathEater_Nmae]
        len_path_deathEater = len(path)
        if len_path_deathEater == 1:
            return [(idx, 1.0)]
        if idx == 0:
            return [(0, 0.5), (1, 0.5)]
        elif idx == len_path_deathEater - 1:
            return [(len_path_deathEater - 1, 0.5), (len_path_deathEater - 2, 0.5)]
        else:
            return [(idx - 1, 1 / 3), (idx, 1 / 3), (idx + 1, 1 / 3)]

    def calculate_reward(self, actions, next_state):
        """
        Calculates the immediate reward for transitioning from the current state
        to the next state based on the given actions.

        Reward criteria:
        - RESET_ACTION results in a penalty of -2 points.
        - DESTROY_ACTION grants +2 points for each destroyed horcrux.
        - If a wizard encounters a death eater in the next state, -1 point is deducted for each encounter.

        :param actions: A list of actions taken by the wizards.
        :param next_state: The resulting state after performing the actions, represented as a tuple:
                           (wizard_positions, death_eater_indices, horcrux_positions, turns_left)
        :return: The calculated reward as an integer.
        """
        reward = 0
        (wizard_positions, death_eater_indices, horcrux_positions, turns_remaining) = next_state

        # Apply penalty for RESET action
        if actions == RESET_ACTION:
            reward += RESET_REWARD

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

    def get_initial_state(self):
        return self.initial_state