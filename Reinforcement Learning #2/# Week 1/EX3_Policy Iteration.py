# Design Policy Iteration ...

import gym
import numpy as np
import sys
env = gym.make('FrozenLake-v0')

def compute_value_function(policy, gamma = 1):
  # Value Evaluation

  Value_Temp = np.zeros((env.nS, env.nA)) # 16 x 4 ... 
  theta = 0.00001

  while True: # Iterate Under Converge Condition
    delta = 0
    for s in range(env.nS): # Each State  ...
      Value = 0  
      a = policy[s] # State s에서의 Action --> Deterministic Policy를 따라 행동 

      for transition_probability, next_state, reward, done in env.P[s][a]:
        next_action = policy[next_state]
        Value += transition_probability * (reward + (gamma*Value_Temp[next_state][next_action]) )

      delta = max(delta, np.abs(Value - Value_Temp[s][a]))
      
      Value_Temp[s][a] = Value

    if delta < theta:
      break    

  value_table = Value_Temp 
  return value_table 

def extract_policy(value_table, gamma = 1.0):
  # Extract Greedy Policy  ... 
  policy = np.argmax(value_table, axis = 1)

  return policy

import random 

def extract_policy_epsilon(value_table, gamma = 1.0):

  policy = np.zeros(env.nS)
  for i in range(env.nS):
    epsilon = np.random.rand()
    if epsilon <= 0.2:
      policy[i] = random.randrange(env.nA)
    else:
      policy[i] = np.argmax(value_table[i], axis = 0)

  policy = policy.astype(np.int)
  return policy 

def policy_iteration(env, gamma = 1.0):
  #random_policy = np.zeros(env.observation_space.n, dtype = np.int32)
  random_policy = np.random.randint(4, size = env.observation_space.n, dtype = np.int32)
  number_of_iteration = 200000

  for i in range(number_of_iteration):
    new_value_function = compute_value_function(random_policy, gamma)
    #new_policy = extract_policy(new_value_function, gamma)
    new_policy = extract_policy_epsilon(new_value_function, gamma)

    if ( np.all(random_policy == new_policy) ): # i.e) V_k+1 = V_k  ...
      print("Policy Iteration Converged at step %d." % (i+1))
      break

    random_policy = new_policy

  return new_policy

optimal_policy = policy_iteration(env, gamma = 0.9)

optimal_policy_int = optimal_policy.astype(np.int)

done = False
state = env.reset()
env.render()

while not done:
  action = optimal_policy_int[state]
  new_state, reward, done, info = env.step(action)

  env.render()
  state = new_state
 
