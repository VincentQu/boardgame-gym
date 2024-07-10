import gymnasium as gym
from gymnasium import spaces

from itertools import combinations
import numpy as np


class CantStopActionSpace(gym.Space):
    def __init__(self):
        # Each action is represented as a tuple
        super().__init__(shape=(), dtype=object)

    def sample(self):
        # Action can be one column (1), two columns (2), or stop (3)
        action_type = np.random.choice([1, 2, 3])

        if action_type == 1:
            return (np.random.randint(2, 13),)

        elif action_type == 2:
            columns = np.random.randint(2, 13, size=(2,))
            return tuple(sorted(columns))

        else:
            return (1,)

    def contains(self, x):
        if not isinstance(x, tuple):
            return False
        if len(x) not in [1, 2]:
            return False
        if x == (1,):
            return True
        return all(2 <= i <= 12 for i in x)


class CantStopEnv(gym.Env):
    def __init__(self, num_players):
        super(CantStopEnv, self).__init__()

        self.num_players = num_players
        self.columns = np.arange(start=2, stop=13)
        self.column_lengths = {
            2: 3,
            3: 5,
            4: 7,
            5: 9,
            6: 11,
            7: 13,
            8: 11,
            9: 9,
            10: 7,
            11: 5,
            12: 3,
        }

        self.action_space = CantStopActionSpace()  # 11 columns + 1 stop action

        self.observation_space = spaces.Dict({
            'player_markers': spaces.Dict({
                p: spaces.Dict({
                    c: gym.spaces.Discrete(l) for c, l in self.column_lengths.items()
                }) for p in range(self.num_players)
            }),
            'tmp_markers': spaces.Dict({
                c: gym.spaces.Discrete(l) for c, l in self.column_lengths.items()
            }),
            'current_player': spaces.Discrete(self.num_players),
            'dice': spaces.Box(low=1, high=6, shape=(4,), dtype=int)
        })

        self.reset()

    def reset(self):
        self.player_marker_positions = {p: {c: None for c in self.columns} for p in range(self.num_players)}
        self.tmp_marker_positions = {c: None for c in self.columns}
        self.current_player = np.random.randint(0, self.num_players, size=1)
        self.dice = self.roll_dice()

        # Return observation and auxiliary information dict
        return self._get_observation(), {}

    def roll_dice(self):
        # Roll 4 dice
        return np.random.randint(1, 7, size=4)

    def _get_observation(self):
        return {
            'player_markers': self.player_marker_positions,
            'temp_markers': self.tmp_marker_positions,
            'current_player': self.current_player,
            'dice': self.dice
        }

    def _get_dice_combinations(self):
        dice = self.dice
        # Return all possible combination of summing up 4 dice into pairs
        combinations = {
            tuple(sorted([dice[0] + dice[1], dice[2] + dice[3]])),
            tuple(sorted([dice[0] + dice[2], dice[1] + dice[3]])),
            tuple(sorted([dice[0] + dice[3], dice[1] + dice[2]]))
        }
        return combinations

    def _get_available_columns(self):
        # Set all columns to available
        available_columns = self.columns.copy()

        for column in self.columns:
            for player in range(self.num_players):
                # If a player marker has reached the final slot on a column, remove this one from the available ones
                if self.player_marker_positions[player][column] == self.column_lengths[column] - 1:
                    available_columns = np.delete(available_columns, np.where(available_columns == column))
                    break

            # If the column has a tmp marker on the final slot, also remove from available ones
            if self.tmp_marker_positions[column] == self.column_lengths[column] - 1:
                available_columns = np.delete(available_columns, np.where(available_columns == column))

        return set(available_columns)

    def _get_possible_moves(self):
        dice_combinations = self._get_dice_combinations()
        available_columns = self._get_available_columns()

        tmp_columns = set([column for column, level in self.tmp_marker_positions.items() if level is not None and level != self.column_lengths[column] - 1])
        complete_tmp_columns = set([column for column, level in self.tmp_marker_positions.items() if level == self.column_lengths[column] - 1])
        free_temp_markers = 3 - len(tmp_columns) - len(complete_tmp_columns)

        possible_moves = []
        for pair in dice_combinations:
            pair_available = tuple(p for p in pair if p in available_columns)
            # pair_available = set(pair) & available_columns
            print(pair, pair_available)

            # If none of the columns of the pair are available, no move can be made using this pair
            if len(pair_available) == 0:
                continue

            # If no more temp markers are available, only columns that are in tmp_columns can be moved on
            elif free_temp_markers == 0:
                if len(set(pair_available) & tmp_columns) > 0:
                    possible_moves.append(tuple(p for p in pair_available if p in tmp_columns))
                continue

            # If enough free tmp markers for each pair (not already in tmp columns) are available, complete pair is possible
            elif len(set(pair_available) - tmp_columns) <= free_temp_markers:
                possible_moves.append(pair_available)
                # continue

            # If only 1 free tmp marker but two columns, then add each individually
            else:
                possible_moves += [tuple([p]) for p in pair_available]

        return possible_moves
