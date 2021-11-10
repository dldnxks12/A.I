import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg') ---- X window system이 아닌 곳에서 주기적으로 graph를 그릴 때 사용하는 코드

# 0 - left terminal state
# 6 - right terminal state
# 1 ~ 5 means each state names A, B, C, D, E, F

VALUES = np.zeros(7)
VALUES[1:6] = 0.5 # Initial Value function 0.5로
VALUES[6] = 1

TRUE_VALUE = np.zeros(7)
TRUE_VALUE[1:6] = np.arange(1,6) / 6
TRUE_VALUE[6] = 1

ACTION_LEFT = 0
ACTION_RIGHT = 1

def monte_carlo(values, alpha = 0.1, batch = False):
    state = 3 # Start at State C
    trajectory = [3] # what is trajectory?  ---- it means episode

    # if end up with left terminal state , all reward = 0
    # if end up with right terminal statem, all reward = 1

    while True:       # np.random.binomial(n, p , size = None) --- 이항 분포에서 표본을 추출 (0 or 1)을 1/2의 확률로 추출

        # Create Episodes ...
        if np.random.binomial(1, 1/2) == ACTION_LEFT: # 0이 뽑히면
            state = -1
        else:
            state += 1
        trajectory.append(state)

        # state가 terminal state로 갈 때
        if state == 6: # right terminal state
            returns = 1.0
            break
        elif state == 0:
            returns = 0.0
            break

    if not batch:  # what is batch ? ....

        # MC Update --- Incremental Update
        for state_ in trajectory[:-1]: # 마지막 state 제외
            # All State를 모두 Update --- Episode가 왼쪽에서 끝났으면 returns = 0  or returns = 1
            values[state_] += alpha*(returns - values[state_])

    return trajectory, [returns] * (len(trajectory) - 1)  # Episode , rewards


def batch_updating(method, episodes, alpha = 0.001):

    runs  = 100
    total_error = np.zeros(episodes)
    for r in range(runs):
        # VALUES : Global Variable
        current_values = np.copy(VALUES) # Update할 꺼니까 Copy 해오자. Deep Copy일 것 -- 메모리 공유 x
        errors = []

        trajectories = []
        rewards = []
        for ep in range(episodes):
            trajectory_ , rewards_ = monte_carlo(current_values, batch = True)

            trajectories.append(trajectory_)
            rewards.append(rewards_)

            while True:
                updates = np.zeros(7)
                for trajectory_, rewards_ in zip(trajectories, rewards):
                    for i in range(0, len(trajectory_) - 1):
                        updates[trajectory_[i]] += rewards_[i] - current_values[trajectory_[i]]
                updates *= alpha

                if np.sum(np.abs(updates)) < 1e-3:
                    break

                current_values += updates

            errors.append(np.sqrt(np.sum(np.power(current_values - TRUE_VALUE, 2))/  5.0))

        total_error += np.asarray(errors)

    total_error /= runs
    return total_error

def figure_6_2():
    episodes = 100 + 1
    mc_erros = batch_updating('MC', episodes)

    plt.plot(mc_erros, label='MC')
    plt.xlabel('episodes')
    plt.ylabel('RMS error')
    plt.legend()
    plt.show()

    plt.savefig('../images/figure_6_2.png')
    plt.close()

if __name__ == '__main__':
    figure_6_2()





