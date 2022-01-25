'''
1. Value function evaluation
2. Value Iteration
3. Q-Value Iteration
'''

import numpy as np

# Policy Evaluation
print("# - Policy Evaluation - #")
P = [[0.58, 0.42],
     [0.66, 0.34]] # state transition matrix

R = [-0.2, -0.18]  # Reward function R(s)
R = np.transpose(R)

V = [0, 0]
V = np.transpose(V)

gamma = 0.9

for k in range(0, 100):
    V = R + gamma*np.matmul(P, V)
    print(V)

# Value Iteration

print("# - Value Iteration - #")

P0 = [[0.5, 0.5], [0.9, 0.1]] # Action = 0 일 때, state transition matrix
P1 = [[0.6, 0.4], [0.3, 0.7]] # Action = 1 일 때, state transition matrix

R0 = [1.5, -0.1]   # [R(0, 0) , R(1, 0)] # Action 0
R1 = [-0.4, -0.3]  # Action 1
R0 = np.transpose(R0)
R1 = np.transpose(R1)
V = [0, 0]
V = np.transpose(V)
gamma = 0.9

for k in range(0, 100):
    # P[s` | 0, 0] or P[s` | 0, 1]

    # state 0에서 ..
    a1 = np.max( [R0[0] + gamma*np.matmul(P0[0] , V) , R1[0] + gamma*np.matmul(P1[0] , V) ])
    # state 1에서 ..
    a2 = np.max( [R0[1] + gamma*np.matmul(P0[1] , V) , R1[1] + gamma*np.matmul(P1[1] , V) ])

    V = np.transpose([a1, a2])

    print(V)

# Q Value Iteration







