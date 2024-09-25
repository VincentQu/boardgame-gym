import random

class CantStopRandomPolicy:
    def __init__(self):
        pass

    def select_action(self, possible_actions):
        return random.choice(possible_actions)