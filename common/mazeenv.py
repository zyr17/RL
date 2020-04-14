import numpy as np
import random
import gym

class MazeEnv(gym.Env):

    def __init__(self):

        self.MAP = [
            "SXE",
            ".X.",
            "..."
        ]
        self.action_space = gym.spaces.Discrete(4)
        #self.ACT = [ 0, 1, 2, 3 ] # left right up down
        self.ACTDELTA = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.INIT_STATE = 0
        def find_start():
            for i in self.MAP:
                for j in i:
                    if j == 'S':
                        return
                    self.INIT_STATE += 1
        find_start()

    def get_init_state(self):
        return self.statenum2state(self.INIT_STATE)

    def statenum2state(self, num):
        return np.array(np.array(range(len(self.MAP) * len(self.MAP[0]))) == num, dtype='float')

    def get_reward(self, state, act):
        state = np.argmax(state)
        x = state // len(self.MAP[0])
        y = state % len(self.MAP[0])
        xx = x + self.ACTDELTA[act][0]
        yy = y + self.ACTDELTA[act][1]
        if xx < 0 or xx >= len(self.MAP) or yy < 0 or yy >= len(self.MAP[0]): # hit border
            return self.statenum2state(x * len(self.MAP[0]) + y), -1
        if self.MAP[xx][yy] == 'X': # hit wall
            return self.statenum2state(x * len(self.MAP[0]) + y), -1
        if self.MAP[xx][yy] == 'E': # enter exit
            return self.statenum2state(xx * len(self.MAP[0]) + yy), 1
        # normal move
        return self.statenum2state(xx * len(self.MAP[0]) + yy), -1

    def is_terminal(self, state):
        state = np.argmax(state)
        x = state // len(self.MAP[0])
        y = state % len(self.MAP[0])
        return self.MAP[x][y] == 'E'

    def reset(self):
        self.state = self.statenum2state(self.INIT_STATE)
        return self.state

    def step(self, action):
        state = np.argmax(self.state)
        #print(state, action)
        x = state // len(self.MAP[0])
        y = state % len(self.MAP[0])
        xx = x + self.ACTDELTA[action][0]
        yy = y + self.ACTDELTA[action][1]
        if xx < 0 or xx >= len(self.MAP) or yy < 0 or yy >= len(self.MAP[0]): # hit border
            state = x * len(self.MAP[0]) + y
            self.state = self.statenum2state(state)
            return self.state, -1, self.is_terminal(self.state), {}
        if self.MAP[xx][yy] == 'X': # hit wall
            state = x * len(self.MAP[0]) + y
            self.state = self.statenum2state(state)
            return self.state, -1, self.is_terminal(self.state), {}
        if self.MAP[xx][yy] == 'E': # enter exit
            state = xx * len(self.MAP[0]) + yy
            self.state = self.statenum2state(state)
            return self.state, 1, self.is_terminal(self.state), {}
        # normal move
        state = xx * len(self.MAP[0]) + yy
        self.state = self.statenum2state(state)
        return self.state, -1, self.is_terminal(self.state), {}