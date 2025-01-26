ids = ["111111111", "222222222"]

import itertools

RESET_REWARD = -2
DESTROY_HORCRUX_REWARD = 2
DEATH_EATER_CATCH_REWARD = -1

class OptimalWizardAgent:
    def __init__(self, initial):
        self.turns_to_go = initial["turns_to_go"]
        self.grid = initial["map"]
        self.wizards = initial["wizards"]
        self.horcruxes = initial["horcrux"]
        self.death_eaters = initial["death_eaters"]

        self.actions = ["move_up", "move_down", "move_left", "move_right", "destroy"]
        self.joint_actions = self.generate_joint_actions()
        self.states = self.generate_all_states()
        self.reduced_states = {self.reduce_state(state): state for state in self.states}
        self.initial_state = self.reduce_state(initial)
        self.transition_matrices = self.build_transition_matrices()
        self.policy = {}

    def generate_joint_actions(self):
        actions_per_player = [self.actions for _ in range(len(self.wizards))]
        return list(itertools.product(*actions_per_player))

    def generate_all_states(self):
        wizard_combinations = itertools.product(self.grid_positions(), repeat=len(self.wizards))
        wizard_states = [
            {name: pos for name, pos in zip(self.wizards.keys(), combination)}
            for combination in wizard_combinations
        ]

        death_eater_states = itertools.product(*[
            range(len(data["path"])) for data in self.death_eaters.values()
        ])
        death_eater_states_named = [
            {name: index for name, index in zip(self.death_eaters.keys(), combination)}
            for combination in death_eater_states
        ]

        horcrux_combinations = itertools.product(*[
            data["possible_locations"] for data in self.horcruxes.values()
        ])
        horcrux_states = [
            {name: loc for name, loc in zip(self.horcruxes.keys(), combination)}
            for combination in horcrux_combinations
        ]

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
        return matrices

    def build_transition_matrix_for_action(self, joint_action):
        transition_matrix = {}
        for reduced_state, original_state in self.reduced_states.items():
            transition_matrix[reduced_state] = {}
            for reduced_next_state, next_original_state in self.reduced_states.items():
                probability = self.calculate_transition_probability(original_state, next_original_state, joint_action)
                if probability > 0:
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

        for wizard, action in zip(self.wizards.keys(), joint_action):
            pos_from = wizards_from[wizard]
            pos_to = wizards_to[wizard]
            probability = self.calculate_wizard_action_probability(pos_from, pos_to, action)
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

    def calculate_wizard_action_probability(self, pos_from, pos_to, action):
        possible_positions = self.get_possible_wizard_positions(pos_from, action)
        return 1 / len(possible_positions) if pos_to in possible_positions else 0

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

    def get_possible_wizard_positions(self, position, action):
        x, y = position
        moves = {
            "move_up": (x - 1, y),
            "move_down": (x + 1, y),
            "move_left": (x, y - 1),
            "move_right": (x, y + 1),
        }
        target_position = moves.get(action, position)
        return [target_position] if target_position in self.grid_positions() else []

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

        actions_for_wizards = []
        wizards = state["wizards"]

        for wizard, action in zip(self.wizards.keys(), chosen_action):
            if action == "destroy":
                horcrux_to_destroy = self.find_destroyable_horcrux(state, wizard)
                if horcrux_to_destroy:
                    actions_for_wizards.append(("destroy", wizard, horcrux_to_destroy))
                else:
                    actions_for_wizards.append(("move", wizard, wizards[wizard]))
            else:
                actions_for_wizards.append((action, wizard))
        self.turns_to_go-=1
        return tuple(actions_for_wizards)

    def find_destroyable_horcrux(self, state, wizard):
        wizards = state["wizards"]
        horcruxes = state["horcruxes"]

        wizard_position = wizards[wizard]
        for horcrux, position in horcruxes.items():
            if wizard_position == position:
                return horcrux
        return None

    def values(self):
        values = {}
        policy = {}

        for reduced_state in self.reduced_states.keys():
            values[(reduced_state, 0)] = 0

        for t in range(1, self.turns_to_go + 1):
            for reduced_state, original_state in self.reduced_states.items():
                action_values = []

                for joint_action in self.transition_matrices.keys():
                    transition_probs = self.transition_matrices[joint_action].get(reduced_state, {})
                    expected_value = sum(
                        prob * values[(next_state, t - 1)]
                        for next_state, prob in transition_probs.items()
                    )

                    penalty = 0
                    for wizard, action in zip(self.wizards.keys(), joint_action):
                        if action != "destroy":
                            wizard_pos = original_state["wizards"][wizard]
                            for death_eater, index in original_state["death_eaters"].items():
                                death_eater_pos = self.death_eaters[death_eater]["path"][index]
                                if wizard_pos == death_eater_pos:
                                    penalty += DEATH_EATER_CATCH_REWARD

                    action_values.append((expected_value + penalty, joint_action))

                reset_value = RESET_REWARD
                if (self.initial_state, t - 1) in values:
                    reset_value += values[(self.initial_state, t - 1)]
                action_values.append((reset_value, "reset"))

                best_action_value, best_action = max(action_values)
                values[(reduced_state, t)] = best_action_value
                policy[(reduced_state, t)] = best_action

        self.policy = policy
        return values, policy



class WizardAgent:
    def __init__(self, initial):
        raise NotImplementedError  


    def act(self, state):
        raise NotImplementedError  
