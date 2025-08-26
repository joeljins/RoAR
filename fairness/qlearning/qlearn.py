import sys
sys.path.append('.')
import random
from typing import List

import numpy as np
from fairness.utils import p, expected

U_PLUS, U_MINUS = 1, -1.1
C_PLUS, C_MINUS = 1, -1
ALPHA = 0.5
SAMPLE_SIZE = 10
MIN = -3
MAX = 3
BUCKETS = 4

ACTIONS = list(range(SAMPLE_SIZE))

class State:
    '''
    The state is represented by x buckets, where each bucket holds all samples in y-z% of the distribtution
    4 buckets, bucket 1 holds 0-25%, bucket 2 holds 25-50%, etc.
    A unique state would be a list of the number of samples in each bucket
    10 Samples : buckets_a[2, 4, 3, 1], buckets_b[1, 3, 5, 1]
    ''' 
    def __init__(self, env, buckets_a : List[int], buckets_b : List[int]):
        self.env = env
        self.buckets_a = buckets_a
        self.buckets_b = buckets_b
        self.mean_diff = np.mean(env.a) - np.mean(env.b)


    # Don't know if needed
    def clone(self):
        return State(self.env, self.buckets_a.copy(), self.buckets_b.copy())

    def reward(self):
        '''
        Higher utility is better
        Penalize for unfairness
        '''
        util_a = self.env.w_a * np.sum(expected(self.env.a, U_PLUS, U_MINUS))
        util_b = self.env.w_b * np.sum(expected(self.env.b, U_PLUS, U_MINUS))
        if self.mean_diff > ALPHA:
            return 0
        return (util_a + util_b)

    def at_end(self) -> bool:
        '''
        mean difference < ALPHA
        '''
        return self.env.steps > 1

    def execute(self, action):
        '''
        Calculate expected change; apply it to values above the threshold(action), get new samples
        Calculate percentiles of the new samples, update buckets

        '''
        delta_A, delta_B = expected(self.env.a, C_PLUS, C_MINUS), expected(self.env.b, C_PLUS, C_MINUS)
        self.env.a = np.where(self.env.a > self.env.a[action], self.env.a + delta_A, self.env.a)
        self.env.b = np.where(self.env.b > self.env.b[action], self.env.b + delta_B, self.env.b)

        q1_a, q2_a, q3_a = np.percentile(self.env.a, [25, 50, 75])
        q1_b, q2_b, q3_b = np.percentile(self.env.b, [25, 50, 75])

        self.buckets_a[0], self.buckets_b[0] = np.sum(self.env.a <= q1_a), np.sum(self.env.b <= q1_b)
        self.buckets_a[1], self.buckets_b[1] = np.sum((self.env.a > q1_a) & (self.env.a <= q2_a)), np.sum((self.env.b > q1_b) & (self.env.b <= q2_b))
        self.buckets_a[2], self.buckets_b[2] = np.sum((self.env.a > q2_a) & (self.env.a <= q3_a)), np.sum((self.env.b > q2_b) & (self.env.b <= q3_b))
        self.buckets_a[3], self.buckets_b[3] = np.sum(self.env.a > q3_a), np.sum(self.env.b > q3_b)

        self.env.steps += 1

        return self

    def __str__(self):
        return f'buckets_a={self.buckets_a}, buckets_b={self.buckets_b})' 


class Env:

    def __init__(self, a : List[float], b: List[float]):
        self.a = a
        self.b = b
        self.w_a = len(a) / (len(a) + len(b))
        self.w_b = 1 - self.w_a
        self.steps = 0

    def random_state(self):
        a = [random.uniform(MIN, MAX) for _ in range(SAMPLE_SIZE)]
        b = [random.uniform(MIN, MAX) for _ in range(SAMPLE_SIZE)]

        q1_a, q2_a, q3_a = np.percentile(a, [25, 50, 75])
        q1_b, q2_b, q3_b = np.percentile(b, [25, 50, 75])

        buckets_a = [0] * BUCKETS
        buckets_b = [0] * BUCKETS

        buckets_a[0], buckets_b[0] = np.sum(a <= q1_a), np.sum(b <= q1_b)
        buckets_a[1], buckets_b[1] = np.sum((a > q1_a) & (a <= q2_a)), np.sum((b > q1_b) & (b <= q2_b))
        buckets_a[2], buckets_b[2] = np.sum((a > q2_a) & (a <= q3_a)), np.sum((b > q2_b) & (b <= q3_b))
        buckets_a[3], buckets_b[3] = np.sum(a > q3_a), np.sum(b > q3_b)

        return State(self, buckets_a, buckets_b)

class QTable:
    def __init__(self, env: Env, actions: List[int]):
        # initialize your q table
        self.env = env
        self.actions = actions
        self.qtable = {}

    def get_q(self, state: State, action: int) -> float:
        # return the value of the q table for the given state, action
        buckets_a = tuple(state.buckets_a)
        buckets_b = tuple(state.buckets_b)
        if (buckets_a, buckets_b) not in self.qtable:
            self.qtable[(buckets_a, buckets_b)] = [0.0] * len(ACTIONS)
        return self.qtable[(buckets_a, buckets_b)][action]

    def get_q_row(self, state: State) -> List[float]:
        # return the row of q table corresponding to the given state
        buckets_a = tuple(state.buckets_a)
        buckets_b = tuple(state.buckets_b)
        if (buckets_a, buckets_b) not in self.qtable:
            self.qtable[(buckets_a, buckets_b)] = [0.0] * len(ACTIONS)
        return self.qtable[(buckets_a, buckets_b)]

    def set_q(self, state: State, action: int, val: float) -> None:
        # set the value of the q table for the given state, action
        buckets_a = tuple(state.buckets_a)
        buckets_b = tuple(state.buckets_b)
        if (buckets_a, buckets_b) not in self.qtable:
            self.qtable[(buckets_a, buckets_b)] = [0.0] * len(ACTIONS)
        self.qtable[(buckets_a, buckets_b)][action] = val

    def learn_episode(self, alpha: float = .10, gamma: float = .90) -> None:
        # from a random initial state,
        state = self.env.random_state()
        # repeat until an end state is reached (thus completing the episode)
        while not state.at_end():
            # consider a random legal action, execute that action,
            action = random.choice(ACTIONS)
            
            # also print the state after each action
            prev_state = state.clone()
            state.execute(action)
            print(state)
            
            # with the given alpha and gamma values,
            # compute the reward, and update the q table for (state, action).
            reward = state.reward()
            max_q = max(self.get_q_row(state))
            sample = state.reward() + gamma * max_q
            new_qvalue = ((1 - alpha) * self.get_q(prev_state, action)) + (alpha * sample)
            self.set_q(prev_state, action, new_qvalue)
            
    def learn(self, episodes, alpha=.10, gamma=.90) -> None:
        # run <episodes> number of episodes for learning with the given alpha and gamma
        for episode in range(episodes):
            self.learn_episode(alpha, gamma)

    def __str__(self) -> str:
        out = ''
        for (buckets_a, buckets_b), qvals in self.qtable.items():
            out += f'State A: {buckets_a}, State B: {buckets_b} => Q-values: '
            out += ', '.join([f'{q:.2f}' for q in qvals]) + '\n'
        return out
    
if __name__ == "__main__":
    random.seed(0)

    mean_a, std_a = 0.5, 1
    mean_b, std_b = 0.0, 1
    
    a = np.random.normal(mean_a, std_a, SAMPLE_SIZE)
    b = np.random.normal(mean_b, std_b, SAMPLE_SIZE)

    env = Env(a,b)
    
    qt = QTable(env, ACTIONS)
    qt.learn(10)  # Fewer episodes for quick testing
    print(qt)

    q1_a, q2_a, q3_a = np.percentile(a, [25, 50, 75])
    q1_b, q2_b, q3_b = np.percentile(b, [25, 50, 75])

    buckets_a = [0] * BUCKETS
    buckets_b = [0] * BUCKETS

    buckets_a[0], buckets_b[0] = np.sum(a <= q1_a), np.sum(b <= q1_b)
    buckets_a[1], buckets_b[1] = np.sum((a > q1_a) & (a <= q2_a)), np.sum((b > q1_b) & (b <= q2_b))
    buckets_a[2], buckets_b[2] = np.sum((a > q2_a) & (a <= q3_a)), np.sum((b > q2_b) & (b <= q3_b))
    buckets_a[3], buckets_b[3] = np.sum(a > q3_a), np.sum(b > q3_b)

    print(qt.get_q_row(State(env, buckets_a, buckets_b)))
    print(a, b)