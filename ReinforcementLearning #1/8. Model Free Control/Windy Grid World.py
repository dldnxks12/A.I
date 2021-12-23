import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#matplotlib.use('Agg') # png file로 만들 수 있다.

####################################################################################  # DP Setting

# Set # DP size
WORLD_HEIGHT = 7
WORLD_WIDTH = 10

# Windy Weights along the rows
WIND = [0,0,0,1,1,1,2,2,1,0]

# Possible Actions at each State
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

# Epsilon Value for E Greedy Policy
EPSILON = 0.1

# Sarse Step Size  ---- alpha value for MC or TD Update
ALPHA = 0.5

# Reward for each Step
REWARD = -1

# Start Point and Goal Point
START = [3, 0]
GOAL = [3, 7]

# Action
ACTIONS = [ACTION_UP,ACTION_DOWN,ACTION_LEFT,ACTION_RIGHT]
####################################################################################

# Action
def Step(state, action):
    i, j = state
    if action == ACTION_UP:
        return [max(i-1-WIND[j], 0), j]
    elif action == ACTION_DOWN:
        return [max( min(i + 1 - WIND[j], WORLD_HEIGHT-1), 0), j] # WIND에 의해 2칸 위로 올라갈 수도 있으니, 외부에 max가 씌워진다.
    elif action == ACTION_LEFT:
        return [max( i - WIND[j], 0), max(j - 1 , 0)]
    elif action == ACTION_RIGHT:
        return [max( i - WIND[j], 0), min(j + 1, WORLD_WIDTH - 1)]
    else:
        assert False


# play for an episode
def episode(q_value):
    time = 0
    state = START

    # Epsilon Greedy Policy for Policy Improvement ...
    if np.random.binomial(1, EPSILON) == 1: # 탐험 모드
        action = np.random.choice(ACTIONS)
    else: # Optimal Action
        values_ = q_value[state[0], state[1], :]  # 해당 state의 4가지 action들을 포함
        action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)]) # max value가 여러 가지라면 그 중에서 randomly choice

    # Try until Goal state
    while state != GOAL:
        next_state = Step(state, action) # S , A -> S' 찾기

        # S' -> A' 구하기 with Greedy Policy
        if np.random.binomial(1, EPSILON) == 1:
            next_action = np.random.choice(ACTIONS)
        else:
            values_ = q_value[next_state[0], next_state[1], :]
            next_action = np.random.choice([action_ for action_ , value_ in enumerate(values_) if value_ == np.max(values_)])

        # Sarsa Update with S, A, R, S', A'
        q_value[state[0], state[1], action] += ALPHA * (REWARD + q_value[next_state[0], next_state[1], next_action] - q_value[state[0], state[1], action])

        state = next_state
        action = next_action
        time += 1
    return time


def figure_6_3():
    q_value = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4)) # rows x cols x actions
    episode_limit = 500

    steps = []
    ep = 0

    while ep < episode_limit:
        steps.append(episode(q_value))
        ep += 1
    steps = np.add.accumulate(steps)

    plt.plot(steps, np.arange(1, len(steps) + 1))
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')
    plt.savefig("./figure_6_3.png")
    plt.close()

    # See Optimla Policy Footage

    optimal_policy = []

    for i in range(0, WORLD_HEIGHT):
        optimal_policy.append([])
        for j in range(0, WORLD_WIDTH):
            if [i,j] == GOAL:
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

    print("Optimal Policy : ")
    for row in optimal_policy:
        print(row)

if __name__ == "__main__":
    figure_6_3()

