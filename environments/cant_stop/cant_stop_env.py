import gymnasium as gym
from gymnasium import spaces

import numpy as np

class CantStopActionSpace(gym.Space):
    def __init__(self):
        # Each action is represented as a tuple
        super().__init__(shape=(), dtype=object)

    def sample(self):
        # Action can be one column (1), two columns (2), or stop (3)
        action_type = np.random.choice([1, 2, 3])

        if action_type == 1:
            return (np.random.randint(2, 13), )

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
            'temp_markers': spaces.Dict({
                c: gym.spaces.Discrete(l) for c, l in self.column_lengths.items()
            }),
            'current_player': spaces.Discrete(self.num_players),
            'dice': spaces.Box(low=1, high=6, shape=(4,), dtype=int)
        })

        self.reset()

    def reset(self):
        self.player_marker_positions = {p: {c: None for c in self.columns} for p in range(self.num_players)}
        self.temp_marker_positions = {c: None for c in self.columns}
        self.current_player = np.random.randint(0, self.num_players, size=1)
        self.dice = self.roll_dice()

        # Return observation and auxiliary information dict
        return self._get_observation(), {}

    def roll_dice(self):
        return np.random.randint(1, 7, size=4)

    def _get_observation(self):
        return {
            'player_markers': self.player_marker_positions,
            'temp_markers': self.temp_marker_positions,
            'current_player': self.current_player,
            'dice': self.dice
        }

    def _get_dice_combinations(self):
        dice = self.dice
        combinations = [
            (dice[0] + dice[1], dice[2] + dice[3]),
            (dice[0] + dice[2], dice[1] + dice[3]),
            (dice[0] + dice[3], dice[1] + dice[2])
        ]
        return combinations