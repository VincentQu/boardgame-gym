from environments.cant_stop import CantStopEnv
from policies.cant_stop import StopAfterNRollsPolicy

from itertools import combinations
from collections import Counter
import pandas as pd

N_MIN = 10
N_MAX = 16

N_SIMULATIONS = 100

unequal_pairs = set(combinations(range(N_MIN, N_MAX+1), 2))
equal_pairs = {(i, i) for i in range(N_MIN, N_MAX+1)}
duels = list(sorted(unequal_pairs.union(equal_pairs)))

results = []
for duel in duels:
    print(f"{duel} ...")
    win_counter = Counter({player: 0 for player in range(2)})
    for _ in range(N_SIMULATIONS):
        env = CantStopEnv(num_players=2)
        policies = [StopAfterNRollsPolicy(n) for n in duel]

        obs, info = env.reset()
        done = False

        while not done:
            current_player = obs['current_player']
            current_policy = policies[current_player]

            possible_actions = env.get_possible_actions()
            action = current_policy.select_action(possible_actions)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        win_counter.update([env.winner])

    p1_win_rate = win_counter[0] / N_SIMULATIONS
    p2_win_rate = win_counter[1] / N_SIMULATIONS

    duel_result = (duel[0], duel[1], p1_win_rate, p2_win_rate)
    results.append(duel_result)

results_df = pd.DataFrame.from_records(results, columns=['p1_n', 'p2_n', 'p1_win_rate', 'p2_win_rate'])
print(results_df)