from environments.cant_stop import CantStopEnv
from policies.cant_stop import StopAfterNRollsPolicy

N = [10, 10]

env = CantStopEnv(num_players=len(N))
policies = [StopAfterNRollsPolicy(n) for n in N]

obs, info = env.reset()
done = False

while not done:
    current_player = obs['current_player']
    current_policy = policies[current_player]

    possible_actions = env.get_possible_actions()
    action = current_policy.select_action(possible_actions)

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.render()
print(f'Winner: Player {env.winner}')

import pickle
with open('cs_stop_after_n_env.pkl', 'wb') as f:
    pickle.dump(env, f)
