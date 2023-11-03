import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MarrakechEnv(gym.Env):
    def __init__(self, num_players=2):
        super(MarrakechEnv, self).__init__()

        # The standard board is 7x7
        self.board_size = 7
        self.num_players = num_players
        self.num_rugs = 15

        # Board:
        # -1 = empty
        # 0, 1, 2, ... = rug of player 0, 1, 2, ...
        self.board = np.full(shape=(self.board_size, self.board_size), fill_value=-1)

        # Assam's position: starting position is the centre of the board
        self.assam_pos = [self.board_size // 2, self.board_size // 2]

        # Assam's direction:
        # 0: North, 1: East, 2: South, 3: West
        self.assam_dir = 0

        # Remaining rugs:
        self.remaining_rugs = [self.num_rugs for p in range(num_players)]