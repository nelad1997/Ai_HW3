ids = ["319000725", "207036711"]

import itertools
import time

RESET_REWARD = -2
DESTROY_HORCRUX_REWARD = 2
DEATH_EATER_CATCH_REWARD = -1
INIT_TIME_LIMIT = 200

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
        # Check if time has exceeded before generating joint actions
        if time.perf_counter() - self.start > INIT_TIME_LIMIT:
            self.t = 0  # No iterations completed
            return

        self.joint_actions = self.generate_joint_actions()

        # Check if time has exceeded before generating states
        if time.perf_counter() - self.start > INIT_TIME_LIMIT:
            self.t = 0
            return

        self.states = self.generate_all_states()

        # Check if time has exceeded before generating reduced states
        if time.perf_counter() - self.start > INIT_TIME_LIMIT:
            self.t = 0
            return

        self.reduced_states = {self.reduce_state(state): state for state in self.states}
        self.initial_state = self.reduce_state(initial)#intial reduced state

        # Check if time has exceeded before building transition matrices
        if time.perf_counter() - self.start > INIT_TIME_LIMIT:
            self.t = 0
            return

        self.transition_matrices = self.build_transition_matrices()

        # Check if time has exceeded before caching possible actions
        if time.perf_counter() - self.start > INIT_TIME_LIMIT:
            self.t = 0
            return

        self.actions_cache = self.cache_all_possible_actions()

        # Run value iteration and track iterations completed
        elapsed_time = time.perf_counter() - self.start
        if elapsed_time > INIT_TIME_LIMIT:
            self.t = 0
        else:
            self.t = self.value_iteration()

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
            # בדיקת זמן לפני כל איטרציה
            elapsed_time = time.perf_counter() - self.start
            if elapsed_time > INIT_TIME_LIMIT:
                return t - 1  # מחזיר את מספר האיטרציות שבוצעו

            for reduced_state, state in self.reduced_states.items():
                max_value = float("-inf")
                best_action = None

                # קבלת פעולות אפשריות מהקאש
                all_possible_actions = actions_cache[reduced_state]
                all_possible_actions.append("reset")
                all_possible_actions.append("terminate")

                for actions in all_possible_actions:
                    # בדיקת זמן לפני חישובים כבדים
                    elapsed_time = time.perf_counter() - self.start
                    if elapsed_time > INIT_TIME_LIMIT:
                        return t - 1  # מחזיר את מספר האיטרציות שבוצעו

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


class WizardAgent(OptimalWizardAgent):
    def __init__(self, initial):
        super().__init__(initial)  # יורש את הבנאי של OptimalWizardAgent

    def act(self, state):
        """
        פעולה המתבצעת בכל שלב של המשחק. אם הגענו לזמן או תור מוגבל,
        מתבססים על הפתרון עד כה ומחזירים את הפעולה המתאימה.
        """
        rewards={}
        key_state = self.reduce_state(state)
        if self.t == self.turns_to_go:#if we expect to get non-positive expected value we terminate or reset the game
            if self.values_regular[(key_state, self.t)]<=0 and self.values_regular[(self.initial_state, self.t-1)]<=0:
                return "terminate"
            elif self.values_regular[(key_state, self.t)]<=0 and self.values_regular[(self.initial_state, self.t-1)]>2: # we make reset if we expect to earn more then 2 points(the price of "reset" action)
                self.turns_to_go -= 1
                return "reset"

        elif self.t >= self.turns_to_go:
            # משתמשים במדיניות האופטימלית שחושבה
            k = self.turns_to_go
            chosen_action = self.policy_regular.get((key_state, k))
            self.turns_to_go -= 1
            return chosen_action
        else:
            # אם יש זמן נוסף, מבצעים חישוב חדש על המצב הנוכחי
            while self.t < self.turns_to_go:
                for reduced_state, state in self.reduced_states.items():
                    rewards[(reduced_state, 0)] = self.Reward(state, None, 0)
                short_state = self.reduced_states[key_state]
                max_value = float("-inf")
                best_action = None
                all_possible_actions = list(self.all_possible_actions(short_state))
                all_possible_actions.append("reset")
                #all_possible_actions.append("terminate")
                for actions in all_possible_actions:
                    current_state_value = self.Reward(state, actions, 1)
                    if actions != "terminate":
                        if actions == "reset":
                            actions_key = "reset"
                        else:
                            actions_key = tuple(action[0] for action in actions)
                        action_matrix = self.transition_matrices[actions_key]
                        transition_Probabilities_from_Curr_State = action_matrix[key_state]
                        for reduced_state_next, prob in transition_Probabilities_from_Curr_State.items():
                            current_state_value += prob * rewards[(reduced_state_next, 0)]
                    if current_state_value > max_value:
                        max_value = current_state_value
                        best_action = actions
                self.turns_to_go -= 1
                return self.process_joint_actions(best_action)

    def value_iteration(self):
        """
        מבצע איטרציה על הערכים ומחשב את המדיניות הטובה ביותר עבור הסוכן הרגיל.
        """


        # אתחול ערכים עבור t=0
        for reduced_state, state in self.reduced_states.items():
            self.values_regular[(reduced_state, 0)] = self.Reward(state, None, 0)

        k=0
        # איטרציה על זמן t
        for t in range(1, self.turns_to_go + 1):
            # בדיקת זמן לפני כל איטרציה
            k+=1
            elapsed_time = time.perf_counter() - self.start
            if elapsed_time > INIT_TIME_LIMIT:
                return t - 1  # מחזיר את מספר האיטרציות שבוצעו

            for reduced_state, state in self.reduced_states.items():
                max_value = float("-inf")
                best_action = None

                # קבלת פעולות אפשריות מהקאש
                all_possible_actions = self.cache_all_possible_actions()[reduced_state]
                all_possible_actions.append("reset")
                all_possible_actions.append("terminate")

                for actions in all_possible_actions:
                    # בדיקת זמן לפני חישובים כבדים
                    elapsed_time = time.perf_counter() - self.start
                    if elapsed_time > INIT_TIME_LIMIT:
                        return t - 1  # מחזיר את מספר האיטרציות שבוצעו

                    current_state_value = self.Reward(state, actions, t)
                    if actions != "terminate":
                        if actions == "reset":
                            actions_key = "reset"
                        else:
                            actions_key = tuple(action[0] for action in actions)
                        action_matrix = self.transition_matrices[actions_key]
                        transition_probabilities = action_matrix[reduced_state]

                        for reduced_state_next, prob in transition_probabilities.items():
                            current_state_value += prob * self.values_regular[(reduced_state_next, t - 1)]

                    if current_state_value > max_value:
                        max_value = current_state_value
                        best_action = actions

                self.values_regular[(reduced_state, t)] = max_value
                self.policy_regular[(reduced_state, t)] = self.process_joint_actions(best_action)

        return k  # מחזיר את מספר האיטרציות שבוצעו

