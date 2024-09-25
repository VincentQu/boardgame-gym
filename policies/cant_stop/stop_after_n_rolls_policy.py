import random

class StopAfterNRollsPolicy:
    def __init__(self, n):
        assert n >= 1, 'n has to be 1 or larger'
        self.n = n
        self.current_rolls = 1 # Starts at 1 because at start of player's turn one dice roll happens by default


    def reset(self):
        self.current_rolls = 1

    def select_action(self, possible_actions):

        # Stop if max number of rolls has been reached
        if self.current_rolls >= self.n:
            self.reset()
            stop_actions = [action for action in possible_actions if action[1] == False]
            return random.choice(stop_actions)

        # Otherwise continue if possible
        continue_actions = [action for action in possible_actions if action[1] == True]
        if continue_actions:
            self.current_rolls += 1
            return random.choice(continue_actions)

        # Else bust
        self.reset()
        return random.choice(possible_actions)
