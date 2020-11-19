import numpy as np

N = 2000
Win = 0

for q in range(N):
    PlayerA, PlayerB = 0, 0
    for p in range(N):
        if np.random.uniform() >= 0.5:
            PlayerA += 1

        if np.random.uniform() >= 0.5:
            PlayerB += 1
    if PlayerA == PlayerB:
        Win += 1

print(float(Win) / N)
