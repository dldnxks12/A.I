import numpy as np
from Env import GraphicDisplay, Env

class PolicyIteration:

    # Initial Setting
    def __init__(self, env):
        # Env.py에서 설정한 # DP Setting
        self.env = env

        # Value function을 2차원 형태로 초기화
        self.value_table = [[0.0]*env.width for _ in range(env.height)]

        # 상 하 좌 우로 이동할 Policy가 동일하게 초기화
        self.policy_table= [ [0.25,0.25,0.25,0.25]*env.width for _ in range(env.height)]

        # Goal Grid Setting
        self.policy_table[2][2] = []

        # Discound Factor r
        self.discound_factor = 0.9

    # Policy Evaluation
    def policy_evaluation(self):

        # 기존 가치함수를 계산한 후 저장할 가치함수 List 생성 및 초기화
        next_value_table = [[0.0]*self.env.width for _ in range(self.env.height)]

        # All state에 대해 Bellman Expectaion Equation 계산
        for state in self.env.get_all_states():
            value = 0.0

            if state == [2,2]: # goal grid
                next_value_table[state[0]][state[1]] = value
                continue

            # 해당 state에서 할 수 있는 모든 Action을 수행하고, 그 값들을 더하기
            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action) # 해당 state에서 action을 취했을 때의 다음 state
                reward = self.env.get_reward(next_state, action) # 다음 state에서 얻을 수 있는 reward (!= return)
                next_value = self.env.get_value(next_state) # 다음 state에서 얻을 수 있는 state value function

                value += (self.get_policy(state)[action] * (reward + self.discound_factor*next_value))

            next_value_table[state[0]][state[1]] = value

        self.value_table = next_value_table # value table Update --- V_k -> V_k+1

    def get_policy(self, state):
        return self.policy_table[state[0]][state[1]]

    def get_value(self, state):
        return self.value_table[state[0]][state[1]]

    def get_action(self, state):
        policy = self.get_policy(state)
        policy = np.array(policy)
        return np.random.choice(4, 1, p = policy)[0] # 4개 중에 1개를 Policy를 따르는 확률로 Choice


    # 현재 Policy를 state value function으로 평가했다.
    # 이제 Greedy Policy Improvement 방법으로 이 Policy를 발전시키자
    def policy_improvement(self):
        next_policy = self.policy_table
        for state in self.env.get_all_states():
            if state == [2,2]:
                continue

            value_list = []
            result = [0.0,0.0,0.0,0.0]

            # All Action 중에서 가장 높은 q function을 찾아 해당하는 action을 선택하자.
            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(next_state)
                next_value = self.get_value(next_state)

                value = reward + (self.discound_factor*next_value)
                value_list.append(value)

            # value_list에 저장된 보상들 중 가장 큰 놈들
            max_idx_list = np.argwhere(value_list == np.nmax(value_list))
            max_idx_list = max_idx_list.flatten().tolist()
            prob = 1 / len(max_idx_list)

            for idx in max_idx_list:
                result[idx] = prob

            next_policy[state[0]][state[1]] = result

        self.policy_table = next_policy



if __name__ == "__main":
    env = Env()
    policy_iteration = PolicyIteration(env)
    grid_world = GraphicDisplay(policy_iteration)
    grid_world.mainloop()


