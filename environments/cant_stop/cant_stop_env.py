import gymnasium as gym
from gymnasium import spaces

import numpy as np


class CantStopActionSpace(gym.Space):
    def __init__(self):
        # Each action is represented as a tuple (columns, continue_flag)
        # columns is itself a tuple of length 0, 1, 2 indicating the columns on which to advance (0 = bust)
        # continue_flag is boolean indicating whether to roll again or stop
        super().__init__(shape=(), dtype=object)

    def sample(self):
        # There are 5 types of possible actions:
        # 1: Advance on two columns and continue
        # 2: Advance on two columns and stop
        # 3: Advance on one column and continue
        # 4: Advance on one column and stop
        # 5: Do not advance (because no combinations are possible) and bust
        action_type = np.random.choice([1, 2, 3, 4, 5])

        if action_type in [1, 2]:
            columns = tuple(sorted(np.random.choice(range(2, 13), size=2)))
            continue_flag = action_type == 1

        elif action_type in [3, 4]:
            columns = (np.random.choice(range(2, 13)),)
            continue_flag = action_type == 3

        else:
            columns = tuple()
            continue_flag = False

        return (columns, continue_flag)


    def contains(self, x):

        if not isinstance(x, tuple) or len(x) != 2:
            return False # x must be tuple of length 2 (columns, continue_flag)

        columns, continue_flag = x
        if not isinstance(columns, tuple) or not isinstance(continue_flag, bool):
            return False # columns must be tuple and continue_flag boolean

        if len(columns) == 0:
            return True # Bust

        if len(columns) > 2:
            return False # 2 columns max

        return all(2 <= c <= 12for c in columns) # Columns must be valid (2-12)


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
        self.dice = self._roll_dice()

        # Return observation and auxiliary information dict
        return self._get_observation(), {}

    def step(self, action):

        assert self.action_space.contains(action)
        columns, continue_flag = action

        if len(columns) == 0: # Bust
            reward = -1 #TODO: Set correct reward value

            # Reset tmp markers
            self.tmp_marker_positions = {c: None for c in self.columns}

            # Set next player and roll dice
            self.current_player = (self.current_player + 1) % self.num_players
            self.dice = self._roll_dice()

            terminated = False
            observation = self._get_observation()

        else: # Move
            reward = self._move_markers(action)

            if continue_flag: # continue
                self.dice = self._roll_dice()
                terminated = False
                observation = self._get_observation()

            else: # stop
                # Update player marker positions and reset tmp markers
                reward = self._end_turn()
                terminated = self._check_game_end()

                if not terminated:
                    # Set next player and roll dice
                    self.current_player = (self.current_player + 1) % self.num_players
                    self.dice = self._roll_dice()

                observation = self._get_observation()

        return observation, reward, terminated, False, {}


    def _roll_dice(self):
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

        tmp_columns = set([column for column, level in self.tmp_marker_positions.items() if
                           level is not None and level != self.column_lengths[column] - 1])
        complete_tmp_columns = set(
            [column for column, level in self.tmp_marker_positions.items() if level == self.column_lengths[column] - 1])
        free_temp_markers = 3 - len(tmp_columns) - len(complete_tmp_columns)

        possible_moves = []
        for pair in dice_combinations:
            pair_available = tuple(p for p in pair if p in available_columns)

            # If none of the columns of the pair are available, no move can be made using this pair
            if len(pair_available) == 0:
                continue

            # If no more temp markers are available, only columns that are in tmp_columns can be moved on
            elif free_temp_markers == 0:
                if len(set(pair_available) & tmp_columns) > 0:
                    possible_moves.append(tuple(p for p in pair_available if p in tmp_columns))

            # If enough free tmp markers for each pair (not already in tmp columns) are available, complete pair is possible
            elif len(set(pair_available) - tmp_columns) <= free_temp_markers:
                possible_moves.append(pair_available)

            # If only 1 free tmp marker but two columns, then add each individually
            else:
                possible_moves += [tuple([p]) for p in pair_available]

        return possible_moves

    def _move_markers(self, action):
        for column in action:
            if self.tmp_marker_positions[column] is None:
                self.tmp_marker_positions[column] = 0
            else:
                self.tmp_marker_positions[column] += 1

        reward = 0 #TODO: Set correct reward value
        return reward

    def _end_turn(self):
        # Set positions of tmp markers to marker positions of current player
        for column, position in self.tmp_marker_positions.items():
            if position is not None:
                self.player_marker_positions[self.current_player][column] = position
        # Reset tmp markers
        self.tmp_marker_positions = {c: None for c in self.columns}

        reward = 0 #TODO: Set correct reward value
        return reward

    def _check_game_end(self):
        # Check if any player has 3 columns complete (i.e., marker positioned at highest position)
        for positions in self.player_marker_positions.values():
            complete_columns = sum([positions[column] == self.column_lengths[column] - 1 for column in self.column_lengths])
            if complete_columns >= 3:
                return True
        return False
