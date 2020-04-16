import numpy as np
import random
import time

EPS = 0.9
ALPHA = 0.1
GAMMA = 0.95
TIMEDELAY = 0.1

MAP = [
    "SXE",
    ".X.",
    "..."
]

ACT = [ 0, 1, 2, 3 ] # left right up down
ACTDELTA = [[0, -1], [0, 1], [-1, 0], [1, 0]]

q_table = np.zeros((len(MAP) * len(MAP[0]), len(ACT)), dtype='float')
step_results = []

INIT_STATE = 0
def find_start():
    for i in MAP:
        for j in i:
            if j == 'S':
                return
            INIT_STATE += 1
find_start()

def get_action(state, eps):
    if random.random() > eps:
        return ACT[random.randint(0, len(ACT) - 1)]
    return np.argmax(q_table[state])

def get_reward(state, act):
    x = state // len(MAP[0])
    y = state % len(MAP[0])
    xx = x + ACTDELTA[act][0]
    yy = y + ACTDELTA[act][1]
    if xx < 0 or xx >= len(MAP) or yy < 0 or yy >= len(MAP[0]): # hit border
        return x * 3 + y, -1
    if MAP[xx][yy] == 'X': # hit wall
        return x * 3 + y, -1
    if MAP[xx][yy] == 'E': # enter exit
        return xx * 3 + yy, 100
    # normal move
    return xx * 3 + yy, -1

def is_terminal(state, epoch, step, print_step = False, print_terminal = True, time_delay = 0):
    x = state // len(MAP[0])
    y = state % len(MAP[0])
    ist = MAP[x][y] == 'E'
    if ist:
        if print_terminal:
            print('Epoch %3d, %4d steps' % (epoch, step))
        step_results.append(step)
    elif print_step:
        p = [x for x in MAP]
        p[x] = p[x][:y] + '*' + p[x][y + 1:]
        print('Step %d:\n%s' % (step, '\n'.join(p)))
        time.sleep(time_delay)
    return ist

def q_learning():
    for epoch in range(200):
        state = INIT_STATE
        action = get_action(state, EPS)
        step = 0
        while True:
            step += 1
            next_s, reward = get_reward(state, action)
            next_a = get_action(next_s, 1)
            ist = is_terminal(next_s, epoch, step)
            if ist:
                q_table[state, action] = reward
            else:
                delta = reward + GAMMA * q_table[next_s, next_a] - q_table[state, action]
                q_table[state, action] += ALPHA * delta
            if ist:
                break
            state = next_s
            action = get_action(state, EPS)

def sarsa():
    for epoch in range(200):
        state = INIT_STATE
        action = get_action(state, EPS)
        step = 0
        while True:
            step += 1
            next_s, reward = get_reward(state, action)
            next_a = get_action(next_s, EPS)
            ist = is_terminal(next_s, epoch, step)
            if ist:
                q_table[state, action] = reward
            else:
                delta = reward + GAMMA * q_table[next_s, next_a] - q_table[state, action]
                q_table[state, action] += ALPHA * delta
            if ist:
                break
            state = next_s
            action = next_a

q_learning()
#sarsa()
print(q_table)
print(step_results)