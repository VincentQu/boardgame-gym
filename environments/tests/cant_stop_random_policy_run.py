from environments.cant_stop import CantStopEnv
from policies.cant_stop import CantStopRandomPolicy

import time

NUM_PLAYERS = 3

env = CantStopEnv(num_players=NUM_PLAYERS)
policies = [CantStopRandomPolicy() for _ in range(NUM_PLAYERS)]

obs, info = env.reset()
done = False

while not done:

    current_player = obs['current_player']
    current_policy = policies[current_player]

    possible_actions = env.get_possible_actions()
    action = current_policy.choose_action(possible_actions=possible_actions)

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # env.render()
    # time.sleep(0.2)

print('Game done')

env.render()

import pickle
with open('cs_completed_game_env.pkl', 'wb') as f:
    pickle.dump(env, f)
