import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set # DP size
WORLD_HEIGHT = 4
WORLD_WIDTH = 12

EPSILON = 0.1
GAMMA = 1
ALPHA = 0.5
REWARD = -1

# Start Point and Goal Point
START = [3, 0]
GOAL = [3, 11]


# Action
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

ACTIONS = [ACTION_UP,ACTION_DOWN,ACTION_LEFT,ACTION_RIGHT]


# Action
def Step(state, action):
    i, j = state
    if action == ACTION_UP:
        next_state = [max(i - 1, 0), j]
    elif action == ACTION_DOWN:
        next_state = [min( i + 1, WORLD_HEIGHT - 1), j]
    elif action == ACTION_LEFT:
        next_state = [i, max(j - 1, 0)]
    elif action == ACTION_RIGHT:
        next_state = [i, min(j + 1, WORLD_WIDTH - 1)]
    else:
        assert False

    reward = -1
    if (action == ACTION_DOWN and i == 2 and 1 <= j <= 10) or (action == ACTION_RIGHT and state == START):
        reward = -100
        next_state = START

    return next_state, reward


def choose_action(state, q_value):
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(ACTIONS)
    else:
        values_ = q_value[state[0], state[1], :]
        return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

def sarsa(q_value, step_size = ALPHA):
    state = START
    action = choose_action(state, q_value) # With Epsilon Greedy
    reward = 0.0

    while state != GOAL:
        next_state, reward = Step(state, q_value) # S, A에서 다음 R, S' 를 관찰
        next_action = choose_action(next_state, q_value) # S'에서 A'를 Get
        reward += reward

        # About TD Target ...
        q_value[next_state[0], next_state[1], action] += step_size*(reward + GAMMA*(q_value[next_state[0], next_state[1], next_action]) - q_value[next_state[0], next_state[1], action])

        state = next_state
        action = next_action

    return reward

def q_learning(q_value, step_size = ALPHA):
    state = START
    reward = 0.0

    while state != GOAL:
        action = choose_action(state, q_value) # Epsilon Greedy로 S -> A Get
        next_state, reward = Step(state, action) # A를 수행해셔 R, S'를 Get

        reward += reward

        # A'은 Greedy Policy로 Get !
        q_value[state[0], state[1], action] += step_size* ( reward + GAMMA*np.max(q_value[next_state[0], next_state[1], :]) -  q_value[state[0], state[1], action])

        state = next_state
    return reward


# print optimal policy
def print_optimal_policy(q_value):
    optimal_policy = []
    for i in range(0, WORLD_HEIGHT):
        optimal_policy.append([])
        for j in range(0, WORLD_WIDTH):
            if [i, j] == GOAL:
                optimal_policy[-1].append('G')
                continue
            bestAction = np.argmax(q_value[i, j, :])
            if bestAction == ACTION_UP:
                optimal_policy[-1].append('U')
            elif bestAction == ACTION_DOWN:
                optimal_policy[-1].append('D')
            elif bestAction == ACTION_LEFT:
                optimal_policy[-1].append('L')
            elif bestAction == ACTION_RIGHT:
                optimal_policy[-1].append('R')
    for row in optimal_policy:
        print(row)

def figure_6_4():
    # episodes of each run
    episodes = 500

    # perform 40 independent runs
    runs = 50

    rewards_sarsa = np.zeros(episodes)
    rewards_q_learning = np.zeros(episodes)
    for r in tqdm(range(runs)):
        q_sarsa = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
        q_q_learning = np.copy(q_sarsa)
        for i in range(0, episodes):
            rewards_sarsa[i] += sarsa(q_sarsa)
            rewards_q_learning[i] += q_learning(q_q_learning)

    # averaging over independt runs
    rewards_sarsa /= runs
    rewards_q_learning /= runs

    # draw reward curves
    plt.plot(rewards_sarsa, label='Sarsa')
    plt.plot(rewards_q_learning, label='Q-Learning')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.ylim([-100, 0])
    plt.legend()

    plt.savefig('./figure_6_4.png')
    plt.close()

    # display optimal policy
    print('Sarsa Optimal Policy:')
    print_optimal_policy(q_sarsa)
    print('Q-Learning Optimal Policy:')
    print_optimal_policy(q_q_learning)