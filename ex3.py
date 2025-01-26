ids = ["111111111", "222222222"]

import itertools

DESTROY_HORCRUX_REWARD = 2
RESET_REWARD = -2
DEATH_EATER_CATCH_REWARD = -1

class OptimalWizardAgent:
    def __init__(self, initial):
        """
        אתחול המודל בהתבסס על מבנה initial נתון.
        :param initial: מילון המתאר את מצב המשחק
        """
        self.turns_to_go = initial["turns_to_go"]
        self.grid = initial["map"]
        self.valid_positions = self.get_valid_positions()  # חישוב מיקומים מותרים

        # ייצוג קומפקטי למצב הנוכחי
        self.current_state = {
            "wizards": {name: data["location"] for name, data in initial["wizards"].items()},
            "horcruxes": {name: data["location"] for name, data in initial["horcrux"].items()},
            "death_eaters": {name: data["index"] for name, data in initial["death_eaters"].items()}
        }

        # עיבוד נתוני הקוסמים
        self.wizards = initial["wizards"]
        self.wizard_count = len(self.wizards)

        # עיבוד נתוני ההורקרוקסים
        self.horcruxes = initial["horcrux"]
        self.horcrux_data = [
            {
                "name": name,
                "possible_locations": data["possible_locations"],
                "prob_change_location": data["prob_change_location"]
            }
            for name, data in self.horcruxes.items()
        ]

        # עיבוד נתוני אוכלי המוות
        self.death_eaters = {#TODO: CHECK IF WE NEED TO UPDATE IT IN ACT FUNCTION
            name: {"path": data["path"], "index": data["index"]}
            for name, data in initial["death_eaters"].items()
        }

        # פעולות אפשריות
        self.actions = ["move_up", "move_down", "move_left", "move_right", "wait", "destroy", "reset", "terminate"]

        # יצירת כל המצבים האפשריים
        self.states = self.generate_all_states()

        # יצירת מילון הסתברויות מעבר לכל פעולה
        self.transition_matrices = self.build_transition_matrices_as_dict()

        self.values,self.policy=self.values() #TODO:complete the values in order to make VI

    def get_valid_positions(self):
        """
        מחשב את כל המיקומים המותרים לפי המפה
        :return: רשימה של מיקומים מותרים
        """
        valid_positions = []
        for x, row in enumerate(self.grid):
            for y, cell in enumerate(row):
                if cell != 'I':  # אם התא לא חסום
                    valid_positions.append((x, y))
        return valid_positions

    def generate_all_states(self):
        """
        יצירת כל המצבים האפשריים על בסיס מיקומים מותרים
        :return: רשימת מצבים
        example for a possible output:
        {
        "wizards": {"Harry": (0, 0), "Ron": (1, 1)},
        "death_eaters": {"Lucius": 0},
        "horcruxes": {"Nagini": (0, 3)}
        },
        {
        "wizards": {"Harry": (0, 0), "Ron": (1, 1)},
        "death_eaters": {"Lucius": 1},
        "horcruxes": {"Nagini": (1, 3)}
        },
        each of them is a possible state
        """
        # כל הקומבינציות של מיקומי קוסמים עם שמות
        wizard_combinations = itertools.product(self.valid_positions, repeat=self.wizard_count)
        wizard_combinations_named = [
            {name: position for name, position in zip(self.wizards.keys(), combination)}
            for combination in wizard_combinations
        ]

        # יצירת הקומבינציות של אינדקסים של אוכלי המוות
        death_eater_combinations = itertools.product(*[
            range(len(data["path"])) for data in self.death_eaters.values()
        ])
        death_eater_combinations_named = [
            {name: index for name, index in zip(self.death_eaters.keys(), combination)}
            for combination in death_eater_combinations
        ]

        # כל הקומבינציות של מיקומי הורקרוקסים עם שמות
        horcrux_combinations = itertools.product(*[data["positions"] for data in self.horcrux_data])
        horcrux_combinations_named = [
            {data["name"]: position for data, position in zip(self.horcrux_data, combination)}
            for combination in horcrux_combinations
        ]

        # כל המצבים האפשריים
        all_states = [
            {"wizards": wizards, "death_eaters": death_eaters, "horcruxes": horcruxes}
            for wizards in wizard_combinations_named
            for death_eaters in death_eater_combinations_named
            for horcruxes in horcrux_combinations_named
        ]
        return all_states

    def build_transition_matrices_as_dict(self):
        """
        בונה מילון של הסתברויות מעבר עבור כל פעולה
        :return: מילון הסתברויות מעבר לכל פעולה
        an example for a possible output:
        {
        "move_up": {
            ("state_from_1_key", "state_from_1_value"): {
            ("state_to_1_key", "state_to_1_value"): probability_1,
            ("state_to_2_key", "state_to_2_value"): probability_2,
            ...
        },
        ...
        },
        "move_down": {
        ...
        },
        ...
        }
        """
        matrices = {}

        for action in self.actions:
            # יצירת מילון הסתברויות לכל פעולה
            action_matrix = {}

            # חישוב הסתברויות מעבר לכל מצב
            for state_from in self.states:
                transition_probs = {}

                # חישוב הסתברויות מעבר ממצב המקור לכל מצב יעד
                for state_to in self.states:
                    probability = self.calculate_transition_probability(state_from, state_to, action)
                    #if probability > 0:  # שומרים רק הסתברויות חיוביות
                    transition_probs[tuple(state_to.items())] = probability

                action_matrix[tuple(state_from.items())] = transition_probs

            matrices[action] = action_matrix

        return matrices

    def calculate_transition_probability(self, state_from, state_to, action):
        """
        מחשב את ההסתברות לעבור ממצב מקור למצב יעד עבור פעולה מסוימת
        :param state_from: מצב מקור
        :param state_to: מצב יעד
        :param action: פעולה
        :return: ההסתברות לעבור ממצב מקור ליעד
        """
        wizard_positions_from, death_eater_positions_from, horcrux_positions_from = (
            state_from["wizards"], state_from["death_eaters"], state_from["horcruxes"]
        )
        wizard_positions_to, death_eater_positions_to, horcrux_positions_to = (
            state_to["wizards"], state_to["death_eaters"], state_to["horcruxes"]
        )

        # הסתברויות מעבר עבור הקוסמים
        wizard_probability = self.calculate_wizard_transition_probability(wizard_positions_from,wizard_positions_to)

        # הסתברויות מעבר עבור אוכלי המוות
        death_eater_probability = self.calculate_death_eater_transition_probability(death_eater_positions_from, death_eater_positions_to)

        # הסתברויות מעבר עבור ההורקרוקסים
        horcrux_probability = self.calculate_horcrux_transition_probability(horcrux_positions_from, horcrux_positions_to)

        # ההסתברות הכוללת היא מכפלת ההסתברויות
        return wizard_probability * death_eater_probability * horcrux_probability

    def calculate_wizard_transition_probability(self, positions_from, positions_to):
        """
        מחשב את הסתברות המעבר עבור הקוסמים
        :param positions_from: מיקומים נוכחיים של הקוסמים (dictionary: {name: current_position})
        :param positions_to: מיקומים עתידיים של הקוסמים (dictionary: {name: next_position})
        :return: 1 אם כל הקוסמים יכולים לעבור למצב העתידי, אחרת 0
        """
        for name, position_from in positions_from.items():
            possible_positions = self.get_possible_wizard_positions(position_from)
            if positions_to[name] not in possible_positions:
                return 0
        return 1

    def get_possible_wizard_positions(self, position):
        """
        מחזיר את כל המיקומים האפשריים שאליהם קוסם יכול לעבור מתא נתון
        :param position: המיקום הנוכחי של הקוסם
        :return: רשימת מיקומים אפשריים
        """
        x, y = position
        possible_positions = []

        # בדיקת כל התאים הסמוכים (למעלה, למטה, שמאלה, ימינה)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_x, new_y = x + dx, y + dy
            if (new_x, new_y) in self.valid_positions:
                possible_positions.append((new_x, new_y))

        return possible_positions

    def calculate_death_eater_transition_probability(self, positions_from, positions_to):
        """
        מחשב את הסתברות מעבר עבור אוכלי המוות
        :param positions_from: אינדקסים נוכחיים של אוכלי המוות במסלול שלהם (dictionary: {name: current_index})
        :param positions_to: אינדקסים עתידיים של אוכלי המוות במסלול שלהם (dictionary: {name: next_index})
        :return: ההסתברות למעבר
        """
        probabilities = []

        for name, index_from in positions_from.items():
            death_eater_data = self.death_eaters[name]  # נתוני המסלול של אוכל המוות הנוכחי
            path = death_eater_data["path"]

            # מיקומים אפשריים: להישאר, לזוז אחורה, לזוז קדימה
            possible_indices = [index_from]  # להישאר במקום
            if index_from > 0:
                possible_indices.append(index_from - 1)  # אחורה
            if index_from < len(path) - 1:
                possible_indices.append(index_from + 1)  # קדימה

            # חישוב הסתברות המעבר
            if positions_to[name] in possible_indices:
                probabilities.append(1 / len(possible_indices))  # הסתברות שווה בין האפשרויות
            else:
                probabilities.append(0)  # לא ניתן לעבור למיקום זה

        # חישוב ההסתברות הכוללת
        probability = 1
        for p in probabilities:
            probability *= p

        return probability

    def calculate_horcrux_transition_probability(self, positions_from, positions_to):
        """
        מחשב את הסתברות מעבר עבור ההורקרוקסים
        :param positions_from: מיקומים נוכחיים של ההורקרוקסים (dictionary: {name: current_position})
        :param positions_to: מיקומים עתידיים של ההורקרוקסים (dictionary: {name: next_position})
        :return: ההסתברות למעבר
        """
        probabilities = []

        for name, position_from in positions_from.items():
            horcrux_info = None
            for h in self.horcrux_data:
                if h["name"] == name:
                    horcrux_info = h
                    break

            if horcrux_info is None:
                probabilities.append(0)
                continue

            possible_locations = horcrux_info["possible_locations"]
            prob_change_location = horcrux_info["prob_change_location"]

            # הסתברות להישאר באותו מיקום
            if positions_to[name] == position_from[name]:
                probabilities.append(1 - prob_change_location + (prob_change_location / len(possible_locations)))
            # הסתברות לעבור למיקום אחר
            elif positions_to[name] in possible_locations:
                probabilities.append(prob_change_location / len(possible_locations))
            else:
                probabilities.append(0)

        # חישוב ההסתברות הכוללת
        probability = 1
        for p in probabilities:
            probability *= p

        return probability

    def act(self, state):
        self.current_state = {
            "wizards": {name: data["location"] for name, data in state["wizards"].items()},
            "horcruxes": {name: data["location"] for name, data in state["horcrux"].items()},
            "death_eaters": {name: data["index"] for name, data in state["death_eaters"].items()}
        }
        key_state_current=tuple(state.items())#transforming the current state to a key for the transition probabilities matrix
        action=self.policy[(key_state_current,self.turns_to_go)]
        self.turns_to_go-=1
        return action

    def values(self):
        """
        מחשבת את הערך המצופה עבור כל מצב ולכל פרק זמן, וגם שומרת את המדיניות האופטימלית.
        """
        values = {}
        policy = {}

        # אתחול הערכים בזמן האחרון (t=0) לתגמול המיידי
        for state in self.states:
            key_state = tuple(state.items())
            values[(key_state, 0)] = self.Reward(state)

        # חישוב הערכים לכל פרק זמן
        for t in range(1, self.turns_to_go + 1):
            for state in self.states:
                key_state = tuple(state.items())  # state from
                action_values = []

                # חישוב ערך עבור כל פעולה אפשרית
                for action in self.actions:
                    transition_probs = self.transition_matrices[action].get(key_state, {})
                    expected_value = 0

                    # חישוב הערך המצופה על בסיס הסתברויות מעבר
                    for state_to, prob in transition_probs.items():
                        key_state_to = (state_to, t - 1)
                        if key_state_to in values:
                            expected_value += prob * values[key_state_to]

                    # התחשבות בפעולת reset
                    if action == "reset":
                        expected_value += RESET_REWARD
                    if action == "move_up"
                    action_values.append((expected_value, action))

                # בחירת הפעולה הטובה ביותר ושמירת הערך והמדיניות
                best_action_value, best_action = max(action_values)
                values[(key_state, t)] = best_action_value + self.Reward(state)
                policy[(key_state, t)] = best_action

        return values, policy

    def Reward(self, state):
        """
        מחשבת את התגמול במצב הנוכחי
        :param state: מצב נתון
        :return: ערך התגמול
        """
        reward = 0

        # תגמול על השמדת הורקרוקסים
        for wizard, wizard_pos in state["wizards"].items():
            for horcrux, horcrux_pos in state["horcruxes"].items():
                if wizard_pos == horcrux_pos:#we assume that if we can destroy horxrox we will do it!
                    reward += DESTROY_HORCRUX_REWARD

        # עונש על מפגש עם אוכלי מוות
        for wizard, wizard_pos in state["wizards"].items():
            for death_eater, death_eater_index in state["death_eaters"].items():
                death_eater_pos = self.death_eaters[death_eater]["path"][death_eater_index]
                if wizard_pos == death_eater_pos:
                    reward += DEATH_EATER_CATCH_REWARD

        return reward


class WizardAgent:
    def __init__(self, initial):
        raise NotImplementedError  


    def act(self, state):
        raise NotImplementedError  
