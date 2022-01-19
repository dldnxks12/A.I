# Find Optimal Policy with Bellman Optimality Equation

import numpy as np
from time import sleep
import gym
import sys

env = gym.make('FrozenLake-v1')
env.render() # Map

def QVI(env, discount_factor = 1.0, theta = 0.0001):
    Q = np.zeros([env.nS, env.nA]) # (16 , 4)

    while True:
        delta = 0

        for s in range(env.nS):  # 각 State 마다 ...
            q = np.zeros(env.nA) # [0, 0, 0, 0]

            for a in range(env.nA): # 각 Action 마다 ...

                # 해당 State에서 해당 Action을 취했을 때, 다음에 나올 State와 그 확률, Reward ...
                for transition_probability, next_state, reward, done in env.P[s][a]:
                    q[a] += transition_probability * (reward + max( Q[next_state] ))

                delta = max(delta, np.abs(q[a] - Q[s][a]))

            # 위에서 계산한 Q value Update
            for i in range(env.nA):
                Q[s][i] = q[i]
        if delta < theta:
            break

    return np.array(Q)


q = QVI(env)

print(f"Q-Function : {q}") # Ok
print("")

# Extract Optimal Policy - Deterministic Policy로 해당 state에서 가장 Value가 높은 놈 Pick
Optimal_Policy = np.argmax(q, axis = 1)

done = False
nA = env.action_space.n
state = env.reset()
env.render()

# Simulate Optimal Policy
Total_reward = 0
Total_Step = 0

while not done:

    new_state, reward, done, info = env.step(Optimal_Policy[state])

    Total_Step += 1
    Total_reward += reward

    state = new_state

    env.render()
    print(f"Current State : {state} || Current Action : {Optimal_Policy[state]}")
    print("")


    sleep(.1)

print("")
print("Total Reward  : ", Total_reward)
print("Total Episode : ", Total_Step)
