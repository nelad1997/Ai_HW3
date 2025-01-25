import random
import networkx as nx
from ex3 import ids, OptimalWizardAgent, WizardAgent
from inputs import inputs
import logging
import time
from copy import deepcopy
from itertools import product


RESET_PENALTY = 2
DESTROY_HORCRUX_REWARD = 2
INIT_TIME_LIMIT = 300
TURN_TIME_LIMIT = 5
DEATH_EATER_PENALTY = 1

WIZARDS = 'wizards'
HORCRUX = 'horcrux'
DEATH_EATERS = 'death_eaters'
TURNS_TO_GO = 'turns_to_go'


def initiate_agent(state):
    """
    initiate the agent with the given state
    """
    if state['optimal']:
        return OptimalWizardAgent(state)

    return WizardAgent(state)


class EndOfGame(Exception):
    """
    Exception to be raised when the game is over
    """
    pass


class WizardStochasticProblem:
    def __init__(self, an_input):
        """
        initiate the problem with the given input
        """
        self.initial_state = deepcopy(an_input)
        self.state = deepcopy(an_input)
        self.graph = self.build_graph()
        start = time.perf_counter()
        self.agent = initiate_agent(deepcopy(self.state))
        end = time.perf_counter()
        if end - start > INIT_TIME_LIMIT:
            logging.critical("timed out on constructor")
            raise TimeoutError
        self.score = 0

    def run_round(self):
        """
        run a round of the game
        """
        while self.state[TURNS_TO_GO] > 0:
            start = time.perf_counter()
            action = self.agent.act(deepcopy(self.state))
            end = time.perf_counter()
            if end - start > TURN_TIME_LIMIT:
                logging.critical(f"timed out on an action")
                raise TimeoutError
            if not self.is_action_legal(action):
                logging.critical(f"You returned an illegal action!")
                print(action)
                raise RuntimeError
            self.result(action)
        self.terminate_execution()

    def is_action_legal(self, action):
        """
        check if the action is legal
        """
        def _is_move_action_legal(move_action):
            wizard_name = move_action[1]
            if wizard_name not in self.state[WIZARDS].keys():
                return False
            l1 = self.state[WIZARDS][wizard_name]["location"]
            l2 = move_action[2]
            return l2 in list(self.graph.neighbors(l1))

        def _is_destroy_action_legal(destroy_action):
            wizard_name = destroy_action[1]
            horcrux_name = destroy_action[2]
            # check near position
            if self.state[WIZARDS][wizard_name]['location'] != self.state[HORCRUX][horcrux_name]['location']:
                return False
            return True
        
        def _is_wait_action_legal(wait_action):
            wizard_name = wait_action[1]
            return wizard_name in self.state[WIZARDS].keys()

        def _is_action_mutex(global_action):
            assert type(
                global_action) == tuple, "global action must be a tuple"
            # one action per wizard
            if len(set([a[1] for a in global_action])) != len(global_action):
                return True
            return False

        if action == "reset":
            return True
        if action == "terminate":
            return True
        
        if len(action) != len(self.state[WIZARDS].keys()):
            logging.error(f"You had given {len(action)} atomic commands, while there are {len(self.state[WIZARDS])}"
                          f" wizards in the problem!")
            return False
        for atomic_action in action:
            # illegal move action
            if atomic_action[0] == 'move':
                if not _is_move_action_legal(atomic_action):
                    logging.error(f"move action {atomic_action} is illegal!")
                    return False
            # illegal destroy action
            elif atomic_action[0] == 'destroy':
                if not _is_destroy_action_legal(atomic_action):
                    logging.error(
                        f"Destroy action {atomic_action} is illegal!")
                    return False
                
            elif atomic_action[0] == 'wait':
                if not _is_wait_action_legal(atomic_action):
                    logging.error(f"Wait action {atomic_action} is illegal!")
                    return False
            else:
                return False
        # check mutex action
        if _is_action_mutex(action):
            logging.error(f"Actions {action} are mutex!")
            return False

        return True

    def result(self, action):
        """"
        update the state according to the action
        """
        self.apply(action)
        if action != "reset":
            self.environment_step()
        self.check_collision_with_death_eaters()

    def apply(self, action):
        """
        apply the action to the state
        """
        if action == "reset":
            self.reset_environment()
            return
        if action == "terminate":
            self.terminate_execution()
        for atomic_action in action:
            self.apply_atomic_action(atomic_action)

    def apply_atomic_action(self, atomic_action):
        """
        apply an atomic action to the state
        """
        wizard_name = atomic_action[1]
        if atomic_action[0] == 'move':
            self.state[WIZARDS][wizard_name]['location'] = atomic_action[2]
            return
        elif atomic_action[0] == 'destroy':
            self.score += DESTROY_HORCRUX_REWARD
            return
        elif atomic_action[0] == 'wait':
            return
        else:
            raise NotImplemented

    def environment_step(self):
        """
        update the state of environment randomly
        """
        for t in self.state[HORCRUX]:
            horcrux_stats = self.state[HORCRUX][t]
            if random.random() < horcrux_stats['prob_change_location']:
                # change destination
                horcrux_stats['location'] = random.choice(
                    horcrux_stats['possible_locations'])

        for death_eater in self.state[DEATH_EATERS]:
            de_stats = self.state[DEATH_EATERS][death_eater]
            index = de_stats["index"]
            if len(de_stats["path"]) == 1:
                continue
            if index == 0:
                de_stats["index"] = random.choice([0, 1])
            elif index == len(de_stats["path"])-1:
                de_stats["index"] = random.choice([index, index-1])
            else:
                de_stats["index"] = random.choice(
                    [index-1, index, index+1])
        self.state[TURNS_TO_GO] -= 1
        return

    def check_collision_with_death_eaters(self):
        for wiz_stats in self.state[WIZARDS].values():
            wiz_loc = wiz_stats["location"]
            for de_stats in self.state[DEATH_EATERS].values():
                index = de_stats["index"]
                de_loc = de_stats["path"][index]
                if wiz_loc == de_loc:
                    self.score -= DEATH_EATER_PENALTY
                    



    def reset_environment(self):
        """
        reset the state of the environment
        """
        self.state[WIZARDS] = deepcopy(
            self.initial_state[WIZARDS])
        self.state[HORCRUX] = deepcopy(self.initial_state[HORCRUX])
        self.state[DEATH_EATERS] = deepcopy(
            self.initial_state[DEATH_EATERS])
        self.state[TURNS_TO_GO] -= 1
        self.score -= RESET_PENALTY
        return

    def terminate_execution(self):
        """
        terminate the execution of the problem
        """
        print(f"End of game, your score is {self.score}!")
        print(f"-----------------------------------")
        raise EndOfGame

    def build_graph(self):
        """
        build the graph of the problem
        """
        n, m = len(self.initial_state['map']), len(
            self.initial_state['map'][0])
        g = nx.grid_graph((m, n))
        nodes_to_remove = []
        for node in g:
            if self.initial_state['map'][node[0]][node[1]] == 'I':
                nodes_to_remove.append(node)
        for node in nodes_to_remove:
            g.remove_node(node)
        return g


def main():
    print(f"IDS: {ids}")
    for an_input in inputs:
        try:
            print(an_input)
            my_problem = WizardStochasticProblem(an_input)
            my_problem.run_round()
        except EndOfGame:
            continue
    for an_input in []: #additional_inputs:
        try:
            my_problem = WizardStochasticProblem(an_input)
            my_problem.run_round()
        except EndOfGame:
            continue


if __name__ == '__main__':
    main()
