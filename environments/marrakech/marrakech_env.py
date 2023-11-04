from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class Direction(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

class MarrakechEnv(gym.Env):
    def __init__(self, num_players=2):
        super(MarrakechEnv, self).__init__()

        # The standard board is 7x7
        self.board_size = 7
        self.max_pos = self.board_size - 1
        self.num_players = num_players
        self.num_rugs = 15

        # Board:
        # -1 = empty
        # 0, 1, 2, ... = rug of player 0, 1, 2, ...
        self.board = np.full(shape=(self.board_size, self.board_size), fill_value=-1)

        # Assam's position: starting position is the centre of the board
        self.assam_pos = [self.board_size // 2, self.board_size // 2]

        # Assam's direction:
        self.assam_dir = Direction.NORTH

        # Remaining rugs:
        self.remaining_rugs = [self.num_rugs for p in range(num_players)]

    def _move_assam(self, die_roll):
        for _ in range(die_roll):
            new_x, new_y = self.assam_pos

            if self.assam_dir == Direction.NORTH: # North (rows -> x)
                new_x -= 1
            if self.assam_dir == Direction.EAST: # East (cols -> y)
                new_y += 1
            if self.assam_dir == Direction.SOUTH: # South
                new_x += 1
            if self.assam_dir == Direction.WEST: # West
                new_y -= 1

            # Assam hit North border
            if new_x < 0:
                new_x = 0
                if new_y == self.max_pos:
                    self.assam_dir = Direction.WEST
                else:
                    # If on even y position, move 1 to the right, else one to the left
                    new_y = new_y + 1 if new_y % 2 == 0 else new_y - 1
                    self.assam_dir = Direction.SOUTH

            # Assam hit East border
            if new_y > self.max_pos:
                new_y = self.max_pos
                if new_x == 0:
                    self.assam_dir = Direction.SOUTH
                else:
                # If on even x position, move 1 to the top, else one to the bottom
                    new_x = new_x - 1 if new_x % 2 == 0 else new_x + 1
                    self.assam_dir = Direction.WEST

            # Assam hit South border
            if new_x > self.max_pos:
                new_x = self.max_pos
                if new_y == 0:
                    self.assam_dir = Direction.EAST
                else:
                    # If on even y position, move 1 to the left, else one to the right
                    new_y = new_y - 1 if new_y % 2 == 0 else new_y + 1
                    self.assam_dir = Direction.NORTH

            # Assam hit West border
            if new_y < 0:
                new_y = 0
                if new_x == self.max_pos:
                    self.assam_dir = Direction.NORTH
                else:
                # If on even x position, move 1 to the bottom, else one to the top
                    new_x = new_x + 1 if new_x % 2 == 0 else new_x - 1
                    self.assam_dir = Direction.EAST

            self.assam_pos = [new_x, new_y]

    def _get_state(self):
        state = {
            'board': self.board,
            'assam_pos': self.assam_pos,
            'assam_dir': self.assam_dir,
            'remaining_rugs': self.remaining_rugs
        }
        return state

    def _roll_die(self):
        values = [1, 2, 3, 4]
        weights = [1/6, 1/3, 1/3, 1/6]
        return np.random.choice(values, p=weights)
