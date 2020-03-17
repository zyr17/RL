import cv2
import gym
import gym.spaces
import numpy as np
import collections
import pdb


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        
        res = np.array(obs).astype(np.float32) / 255.0
        #pdb.set_trace()
        return res


class MultiLives(gym.ObservationWrapper):
    def __init__(self, env):
        super(MultiLives, self).__init__(env)
        self.env = env
        self.lives = 0
        self.real_over = True
    def reset(self):
        if (self.real_over):
            state = self.env.reset()
            self.real_over = False
        else:
            state, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return state
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.real_over = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

class SkipFrame(gym.ObservationWrapper):
    def __init__(self, env, skip_number = 4):
        super(SkipFrame, self).__init__(env)
        self.SKIP = skip_number
    def reset(self):
        state = self.env.reset()
        self.COUNTER = self.SKIP
        return state
    def step(self, action):
        total_reward = 0
        states = []
        for i in range(self.SKIP):
            state, reward, terminal, _ = self.env.step(action)
            states.append(state)
            total_reward += reward
            if terminal:
                break
        return np.max(np.stack(states), axis = 0), total_reward, terminal, _

class ResizeGreyPic(gym.ObservationWrapper):
    def __init__(self, env, cut = [-1, -1], size = (84, 84)):
        super(ResizeGreyPic, self).__init__(env)
        self.SIZE = size
        self.observation_space = gym.spaces.Box(low = 0, high = 255, shape = (size[1], size[0], 1), dtype=np.float)
        self.cut = cut

    def observation(self, state):
        #print(state.shape)
        #pdb.set_trace()
        if len(state.shape) == 3:
            state = state.reshape(1, *state.shape)
        new_s_all = []
        for s in state:
            new_s = s[:,:,0] * 0.299 + s[:,:,1] * 0.587 + s[:,:,2] * 0.114
            if self.cut[0] != -1 or self.cut[1] != -1:
                new_s = new_s[self.cut[0]:self.cut[1], :]
                #new_s = new_s[34:194, :]#Pong
                #new_s = new_s[32:192, :]#Breakout
            new_s_all.append(new_s)
        #pdb.set_trace()
        new_s_all = np.stack(new_s_all).max(0)
        #print(new_s_all.shape)
        state = cv2.resize(new_s_all, self.SIZE, interpolation = cv2.INTER_AREA)
        state = state.reshape((*self.SIZE, 1)).astype(np.uint8)
        #plt.matshow(state)
        #delta = x_t - state
        #pdb.set_trace()
        #plt.show()
        return state# / 255.0

class ChangeAxis(gym.ObservationWrapper):
    def __init__(self, env):
        super(ChangeAxis, self).__init__(env)
        size = self.env.observation_space.shape
        self.observation_space = gym.spaces.Box(low = self.env.observation_space.low, high = self.env.observation_space.high, shape = (size[2], size[0], size[1]), dtype=self.env.observation_space.dtype)
    def observation(self, state):
        return state.transpose(2, 0, 1)

class CollectFrame(gym.ObservationWrapper):
    def __init__(self, env, collect_number = 4):
        super(CollectFrame, self).__init__(env)
        self.COLLECT = collect_number
        #pdb.set_trace()
        oldshape = self.env.observation_space.shape
        if len(oldshape) == 3:
            if oldshape[0] == 1: oldshape = (oldshape[1], oldshape[2])
            elif oldshape[2] == 1: oldshape = (oldshape[0], oldshape[1])
        #print(oldshape)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(collect_number, *oldshape),
                                                dtype=self.env.observation_space.dtype)
        #pdb.set_trace()
    def reset(self):
        oldshape = self.env.observation_space.shape
        if len(oldshape) == 3:
            if oldshape[0] == 1: oldshape = (oldshape[1], oldshape[2])
            elif oldshape[2] == 1: oldshape = (oldshape[0], oldshape[1])
        self.states = np.zeros((self.COLLECT, *oldshape), dtype='float32')
        return self.observation(self.env.reset())
    def observation(self, state):
        if len(state.shape) == 3 and state.shape[0] == 1: state = state.reshape(state.shape[1], state.shape[2])
        if len(state.shape) == 3 and state.shape[2] == 1: state = state.reshape(state.shape[0], state.shape[1])
        self.states[:-1] = self.states[1:]
        self.states[-1] = state
        #pdb.set_trace()
        return self.states

def make_env(env_name):
    '''
    env = gym.make("PongNoFrameskip-v4")
    env = SkipFrame(env)
    env = ResizeGreyPic(env)
    #env = ChangeAxis(env)
    env = CollectFrame(env)
    return env
    '''
    cut = [-1, -1]
    if env_name[:4] == 'Pong':
        cut = [34, 194]
    elif env_name[:8] == 'Breakout':
        cut = [32, 192]
    env = gym.make(env_name)
    env = MultiLives(env)
    env = SkipFrame(env)
    env = ResizeGreyPic(env, cut)
    env = CollectFrame(env)
    env = ScaledFloatFrame(env)
    return env
