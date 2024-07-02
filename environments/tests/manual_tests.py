from environments.marrakech import MarrakechEnv
from collections import Counter

max_pos = 6
marrakech = MarrakechEnv()

marrakech.assam_pos = [0, 0]

candidates = marrakech.get_rug_pos_candidates()

# for can in candidates:
#     print(can)
# print(candidates)
filtered = [
    candidate for candidate in candidates
    if all(0 <= coord <= max_pos for point in candidate for coord in point)
]


import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))
for pair in filtered:
    p1, p2 = pair

    x_vals, y_vals = list(zip(p1, p2))

    plt.plot(x_vals, y_vals, marker='o', linestyle='-')
plt.grid(True)
plt.show()


rug_placements = [
    {'turn': 0, 'player': 1, 'rug_pos': ([2, 4], [2, 5])},
    {'turn': 1, 'player': 2, 'rug_pos': ([0, 0], [1, 0])},
    {'turn': 2, 'player': 3, 'rug_pos': ([6, 2], [5, 2])},
    {'turn': 3, 'player': 1, 'rug_pos': ([1, 3], [1, 4])}
]