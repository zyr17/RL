import cv2
import gym
import gym.spaces
import numpy as np
import collections
import pdb


class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution. %s" % str(frame.shape)
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        img = img[34:194]
        resized_screen = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        #x_t = resized_screen[18:102, :]
        x_t = np.reshape(resized_screen, [84, 84, 1])
        return x_t.astype(np.uint8)


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        
        res = np.array(obs).astype(np.float32) / 255.0
        #pdb.set_trace()
        return res


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer

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
    def __init__(self, env, size = (84, 84)):
        super(ResizeGreyPic, self).__init__(env)
        self.SIZE = size
        self.observation_space = gym.spaces.Box(low = 0, high = 255, shape = (size[1], size[0], 1), dtype=np.float)
    def observation(self, state):
        #print(state.shape)
        #pdb.set_trace()
        if len(state.shape) == 3:
            state = state.reshape(1, *state.shape)
        new_s_all = []
        for s in state:
            new_s = s[:,:,0] * 0.299 + s[:,:,1] * 0.587 + s[:,:,2] * 0.114
            new_s = new_s[34:194, :]
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
        print(oldshape)
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
    env = gym.make(env_name)
    #env = MaxAndSkipEnv(env)
    env = SkipFrame(env)
    #env = FireResetEnv(env)
    #env = ProcessFrame84(env)
    env = ResizeGreyPic(env)
    #env = ImageToPyTorch(env)
    #env = BufferWrapper(env, 4)
    env = CollectFrame(env)
    env = ScaledFloatFrame(env)
    return env
