import gymnasium as gym
from gymnasium import spaces

import colorama
from colorama import Fore, Back, Style
from IPython.display import display, HTML
import numpy as np
import os
import sys


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

        return all(2 <= c <= 12 for c in columns) # Columns must be valid (2-12)


class CantStopEnv(gym.Env):
    def __init__(self, num_players):
        super(CantStopEnv, self).__init__()

        self.player_colors = ['red', 'green', 'blue', 'yellow']
        self.is_notebook = 'ipykernel' in sys.modules

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
        self.current_player = int(np.random.randint(0, self.num_players, size=1))
        self.dice = self._roll_dice()

        # Return observation and auxiliary information dict
        return self._get_observation(), {}

    def step(self, action):

        assert self.action_space.contains(action), 'Action needs to be a tuple (columns, continue_flag)'
        columns, continue_flag = action

        possible_moves = self._get_possible_moves()
        assert columns in possible_moves, 'Impossible move'

        if len(columns) == 0: # Bust
            assert continue_flag == False, 'Cannot continue after busting'
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


    def render(self):

        colorama.init(autoreset=True)

        # Unicode characters for markers
        PLAYER_MARKER = '●'
        TEMP_MARKER = '○'
        EMPTY_SPACE = '·'
        INACCESSIBLE = '—'

        # Create the board
        board = [[EMPTY_SPACE for _ in range(11)] for _ in range(13)]

        # Fill in inaccessible spaces
        for col, length in self.column_lengths.items():
            for row in range(length, 13):
                board[row][col-2] = INACCESSIBLE

        if self.is_notebook: # Create HTML output
            # Place player markers
            for player, positions in self.player_marker_positions.items():
                for col, pos in positions.items():
                    if pos is not None:
                        # Add player marker (multiple markers possible in same position)
                        board[pos][col-2] += f'<span style="color:{self.player_colors[player]};">{PLAYER_MARKER}</span>'
                        # Remove empty space marker
                        board[pos][col - 2] = board[pos][col-2].replace(EMPTY_SPACE, '')

            # Place temporary markers
            for col, pos in self.tmp_marker_positions.items():
                if pos is not None:
                    # Add tmp marker (multiple markers possible in same position)
                    board[pos][col-2] += f'<span style="color:gray;">{TEMP_MARKER}</span>'
                    # Remove empty space marker
                    board[pos][col - 2] = board[pos][col - 2].replace(EMPTY_SPACE, '')

            # Create the HTML string
            html = '<pre style="line-height: 1.2; font-family: monospace; font-size: 16px;">'
            for row in range(12, -1, -1):
                html += f"{row:2d} "
                for col in range(11):
                    html_cell = board[row][col]
                    items = 0
                    items += html_cell.count(EMPTY_SPACE)
                    items += html_cell.count(INACCESSIBLE)
                    items += html_cell.count('<span')
                    # print(items, html_cell)
                    if items == 1:
                        html += f"  {board[row][col]}  "
                    elif items == 2:
                        html += f" {board[row][col]}  "
                    elif items == 3:
                        html += f" {board[row][col]} "
                    elif items == 4:
                        html += f"{board[row][col]} "
                    else:
                        html += f"{board[row][col]}"
                html += '<br>'

            # Add column numbers (aligned)
            html += "   "  # Extra space to align with board
            for col in range(2, 13):
                html += f"{col:^5}"
            html += '<br><br>'

            # Add current player and dice roll
            html += f"Current player: <span style='color:{self.player_colors[self.current_player]};'>Player {self.current_player}</span><br>"
            html += f"Dice roll: {self.dice}<br>"
            html += '</pre>'

            display(HTML(html))

        else: # Print coloured output using colorama

            color_map = {
                'red': Fore.RED,
                'green': Fore.GREEN,
                'blue': Fore.BLUE,
                'yellow': Fore.YELLOW
            }

            # Place player markers
            for player, positions in self.player_marker_positions.items():
                for col, pos in positions.items():
                    if pos is not None:
                        # Add player marker (multiple markers possible in same position)
                        board[pos][col - 2] = color_map[self.player_colors[player]] + PLAYER_MARKER + Style.RESET_ALL

            # Place temporary markers
            for col, pos in self.tmp_marker_positions.items():
                if pos is not None:
                    # Add tmp marker (multiple markers possible in same position)
                    board[pos][col - 2] = Fore.LIGHTBLACK_EX + TEMP_MARKER + Style.RESET_ALL

            # Print the board
            for row in range(12, -1, -1):
                print(f"{row:2d} ", end="")
                for col in range(11):
                    print(f"  {board[row][col]}  ", end="")
                print()

            # Add column numbers (aligned)
            print("   ", end="")  # Extra space to align with board
            for col in range(2, 13):
                print(f"{col:^5}", end="")
            print("\n")

            # Add current player and dice roll
            print(
                f"Current player: {color_map[self.player_colors[self.current_player]]}Player {self.current_player}{Style.RESET_ALL}")
            print(f"Dice roll: {self.dice}")

    def get_possible_actions(self):
        possible_actions = []
        possible_moves = self._get_possible_moves()

        for move in possible_moves:
            if len(move) == 0:
                # If you bust, you cannot continue
                possible_actions.append((tuple(), False))
            else:
                # If you have a valid move, you can always choose to continue or not
                possible_actions.append((move, True))
                possible_actions.append((move, False))

        return possible_actions

    def _roll_dice(self):
        # Roll 4 dice
        return sorted(np.random.randint(1, 7, size=4))

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

        # Add empty tuple (representing bust) if no moves are possible
        if len(possible_moves) == 0:
            possible_moves.append(tuple())

        return possible_moves

    def _move_markers(self, action):
        columns, _ = action
        for column in columns:
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