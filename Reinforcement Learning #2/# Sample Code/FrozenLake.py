'''
Value Iteration for PE in Frozen Lake - V0
Movement Direction of Agent is Uncertain ( Policy is not Optimal ... just random ! )

# Map

        SFFF
        FHFH
        FFFH
        HFFG

# Action Space = 4

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# State Space = 4

    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located

You receive a reward of 1 if you reach the goal, and zero otherwise.

'''

import gym
from time import sleep
import sys

env = gym.make('FrozenLake-v1') # Frozon Lake Environment
env = gym.wrappers.TimeLimit(env, max_episode_steps = 5) # 시간 제한 걸기
env.reset()

done = False # 현재 State = Goal or Hole 이면 Donw = True 로 ..
env.render() # 그리기

while not done:
    new_state, reward, done, info = env.step(env.action_space.sample())

    print(reward, new_state)

    env.render()
    sleep(.1)






