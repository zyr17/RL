import torch
import numpy as np
import random
import pickle
import mazeenv
import gym
import time
import collections
import cv2
import matplotlib.pyplot as plt

ENABLE_CUDA = True

def cuda(tensor):
    """
    A cuda wrapper
    """
    if tensor is None:
        return None
    if torch.cuda.is_available() and ENABLE_CUDA:
        return tensor.cuda()
    else:
        return tensor

def showarray(arr):
    arr = np.array(arr)
    print('max: %.2f, min: %.2f' % (arr.max(), arr.min()))
    plt.imshow(arr)
    plt.show()

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
        return states[-1], total_reward, terminal, _
                

class ResizeGreyPic(gym.ObservationWrapper):
    def __init__(self, env, size = (84, 110)):
        super(ResizeGreyPic, self).__init__(env)
        self.SIZE = size
        self.observation_space = gym.spaces.Box(low = 0, high = 255, shape = (size[1], size[0]), dtype=np.uint8)
    def observation(self, state):
        new_s = state[:,:,0] * 0.299 + state[:,:,1] * 0.587 +state[:,:,2] * 0.114
        state = cv2.resize(new_s, self.SIZE, interpolation = cv2.INTER_AREA)
        return state / 255.0

class ChangeAxis(gym.ObservationWrapper):
    def __init__(self, env):
        super(ChangeAxis, self).__init__(env)
        size = self.env.observation_space.shape
        self.observation_space = gym.spaces.Box(low = 0, high = 255, shape = (size[2], size[0], size[1]), dtype=np.uint8)
    def observation(self, state):
        return state.transpose(2, 0, 1)

class CollectFrame(gym.ObservationWrapper):
    def __init__(self, env, collect_number = 4):
        super(CollectFrame, self).__init__(env)
        self.COLLECT = collect_number
    def reset(self):
        self.states = np.zeros((self.COLLECT, *self.env.observation_space.shape), dtype='float')
        return self.observation(self.env.reset())
    def observation(self, state):
        self.states[:-1] = self.states[1:]
        self.states[-1] = state
        return self.states

#input: state; output: for every action: q values
class DQNnet(torch.nn.Module):
    def __init__(self, inputlen, cnn, fc):
        super(DQNnet, self).__init__()
        self.const = {}
        self.const['inputlen'] = inputlen
        self.const['cnn'] = cnn
        self.const['fc'] = fc
        self.cnn = torch.nn.ModuleList()
        lastfea = inputlen
        for i in cnn:
            self.cnn.append(torch.nn.Sequential(
                torch.nn.Conv2d(lastfea, i[0], i[1], padding=i[2], stride = i[3]),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(i[4], padding=i[5])
            ))
            #self.cnn[-1].weight.data.normal_(0, 0.1)
            lastfea = i[0]
        self.fc = torch.nn.ModuleList()
        for i, j in zip(fc[:-1], fc[1:]):
            self.fc.append(torch.nn.Linear(i, j))
            self.fc[-1].weight.data.normal_(0, 0.1)
            #torch.nn.Dropout(0.5),
            self.fc.append(torch.nn.ReLU())
        self.fc = self.fc[:-1]

    def forward(self, inputs):
        x = inputs
        for cnn in self.cnn:
            x = cnn(x)
            #print(x.shape)
        x = x.reshape(x.shape[0], -1)
        for fc in self.fc:
            x = fc(x)
            #print(x.shape)
        return x

class DQN:
    def __init__(self, env, inputlen, cnn, fc, 
                 alpha = 0.1, gamma = 0.95, eps = 0.1, 
                 epoch = 1000, replay = 1000000, update_round = 10000,
                 learning_rate = 0.001, batch_size = 128, reward_func = None,
                 render = -1
                ):
        self.env = env
        self.ALPHA = alpha
        self.GAMMA = gamma
        if type(eps) == type(()) or type(eps) == type([]):
            self.EPS = eps[0]
            self.EPS_STEP = eps[1]
            self.EPS_MIN = eps[2]
        else:
            self.EPS = self.EPS_END = eps
            self.EPS_STEP = 0
            self.EPS_MIN = eps
        self.EPOCH = epoch
        self.REPLAY = replay
        self.UPDATE = update_round
        self.model_old = cuda(DQNnet(inputlen, cnn, fc))
        self.model_update = cuda(DQNnet(inputlen, cnn, fc))
        self.model_old.load_state_dict(self.model_update.state_dict())
        self.update_count = 0
        self.replay_count = 0
        self.replay_data = collections.deque(maxlen = replay)
        self.replay_state = []
        self.replay_action = []
        self.replay_reward = []
        self.replay_next_s = []
        self.replay_ist = []
        self.LR = learning_rate
        self.BATCH_SIZE = batch_size
        self.opt = torch.optim.Adam(self.model_update.parameters(), learning_rate)
        self.loss = torch.nn.MSELoss()
        self.model_old.eval()
        self.model_update.train()
        self.reward_func = reward_func
        if type(reward_func) == type(None):
            self.reward_func = lambda env, state, reward: reward
        self.render = render

    def get_action(self, state):
        if random.random() < self.EPS:
            return self.env.action_space.sample()
        q = self.model_update(cuda(torch.tensor([state]).float()))[0]
        if False and self.REPLAY == len(self.replay_data):
            print(q, ' choice:', torch.argmax(q).item(), ' ', end = '')
            showarray(state[-1])
        return torch.argmax(q).item()

    def real_update_q(self, state, action, reward, next_s, ist):
        state = np.array(state)
        action = np.array(action)
        reward = np.array(reward)
        next_s = np.array(next_s)
        ist = np.array(ist)
        #print('real update q', state, action, reward)
        self.opt.zero_grad()
        state = cuda(torch.tensor(state).float())
        action = cuda(torch.tensor(action).long())
        reward = cuda(torch.tensor(reward).float())
        next_s = cuda(torch.tensor(next_s).float())
        q = self.model_update(state)
        action = action.reshape(action.shape[0], 1)
        next_q = self.model_old(next_s).max(dim = 1)[0]
        reward_b = reward + self.GAMMA * next_q * cuda(torch.tensor(1 - ist).float())
        reward_b = reward_b.reshape(reward_b.shape[0], 1)
        L = self.loss(q.gather(1, action), reward_b)
        L.backward()
        self.opt.step()

    def update_q(self, state, action, reward, next_s, ist):
        '''
        if len(self.replay_state) < self.REPLAY:
            #self.replay.append([state, action, reward])
            self.replay_state.append(state)
            self.replay_action.append(action)
            self.replay_reward.append(reward)
            self.replay_next_s.append(next_s)
            self.replay_ist.append(ist)
            if len(self.replay_state) == self.REPLAY:
                self.replay_state = np.array(self.replay_state)
                self.replay_action = torch.tensor(self.replay_action).long()
                self.replay_reward = torch.tensor(self.replay_reward).float()
                self.replay_next_s = np.array(self.replay_next_s)
                self.replay_ist = np.array(self.replay_ist)
                #print(self.replay_state.dtype, self.replay_action.dtype, self.replay_reward.dtype)
        else:
            self.replay_count = (self.replay_count + 1) % self.REPLAY
            self.replay_state[self.replay_count] = state
            self.replay_action[self.replay_count] = action
            self.replay_reward[self.replay_count] = reward
            self.replay_next_s[self.replay_count] = next_s
            self.replay_ist[self.replay_count] = ist
            choice = np.random.permutation(self.REPLAY)[:self.BATCH_SIZE]
            self.real_update_q(self.replay_state[choice], self.replay_action[choice], self.replay_reward[choice], self.replay_next_s[choice], self.replay_ist[choice])
            #self.real_update_q(*self.replay[self.replay_count])
            self.update_count = (self.update_count + 1) % self.UPDATE
            if self.update_count == 0:
                #print('update model')
                self.model_old.load_state_dict(self.model_update.state_dict())
        '''
        self.replay_data.append([state, action, reward, next_s, ist])
        if len(self.replay_data) == self.REPLAY:
            choice = np.random.choice(self.REPLAY, self.BATCH_SIZE, replace = False)
            self.real_update_q(*zip(*[self.replay_data[x] for x in choice]))
            self.update_count = (self.update_count + 1) % self.UPDATE
            if self.update_count == 0:
                self.model_old.load_state_dict(self.model_update.state_dict())
    
    def sampling(self, epoch):
        start_time = time.time()
        state = self.env.reset()
        init_state = state
        action = self.get_action(state)
        step = 0
        tot_reward = 0
        while True:
            step += 1
            if self.render != -1:
                self.env.render()
                time.sleep(self.render)
            next_s, reward, ist, _ = self.env.step(action)
            reward = self.reward_func(self.env, next_s, reward)
            tot_reward += reward
            self.update_q(state, action, torch.tensor(reward).float(), next_s, 1 if ist else 0)
            if ist:
                print('Epoch %6d, %6d steps, %.2f steps/s' % (epoch, step, step / (time.time() - start_time)), tot_reward)
                #print(self.model_update(torch.tensor([init_state]).float())[0])
                '''
                for i in range(9):
                    i = torch.tensor(np.array(np.array(range(9)) == i, dtype='float')).float()
                    print(self.model_update(i))
                '''
                break
            state = next_s
            action = self.get_action(state)
    
    def main(self):
        for ep in range(self.EPOCH):
            self.sampling(ep)
            self.EPS = max(self.EPS - self.EPS_STEP, self.EPS_MIN)
'''      
# MazeEnv
inputlen = 9
cnn = []
fc = [9, 100, 10, 4]
env = mazeenv.MazeEnv()
dqn = DQN(env, inputlen, cnn, fc, gamma = 0.9, learning_rate = 0.01,
          epoch = 100000, replay = 2000, update_round = 100)
'''
'''
# CartPole
inputlen = 4
cnn = []
fc = [4, 100, 10, 2]
env = gym.make("CartPole-v0")
env = env.unwrapped
def CartPole_reward_func(env, state, reward):
    x = state[0]
    theta = state[2]
    r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.5
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    reward = float(r1 + r2)
    return reward
dqn = DQN(env, inputlen, cnn, fc, gamma = 0.9, learning_rate = 0.01,
          epoch = 100000, replay = 2000, update_round = 100, reward_func=CartPole_reward_func)
'''
'''
#MsPacman RAM
inputlen = 128
cnn = []
fc = [128, 1000, 100, 9]
env = gym.make("MsPacman-ram-v0")
env = env.unwrapped
dqn = DQN(env, inputlen, cnn, fc, gamma = 0.9, learning_rate = 0.0001,
          epoch = 100000, replay = 10000, update_round = 1000, render = 0)
'''
'''
#Pong RAM
inputlen = 128
cnn = []
fc = [128, 1000, 100, 6]
env = gym.make("Pong-ram-v4")
env = env.unwrapped
dqn = DQN(env, inputlen, cnn, fc, gamma = 0.9, learning_rate = 0.0001,
          epoch = 100000, replay = 10000, update_round = 1000, render = 0)
'''

#Pong CNN
inputlen = 4
cnn = [
    (32, 8, 0, 4, 1, 0),
    (64, 4, 0, 2, 1, 0),
    (64, 3, 0, 1, 1, 0),
]
fc = [7 * 10 * 64, 256, 6]
env = gym.make("PongNoFrameskip-v4")
env = SkipFrame(env)
env = ResizeGreyPic(env)
#env = ChangeAxis(env)
env = CollectFrame(env)
dqn = DQN(env, inputlen, cnn, fc, gamma = 0.9, learning_rate = 0.0001, eps = [0.95, 0.01, 0.01],
          epoch = 100000, replay = 10000, update_round = 1000, render = -1, batch_size = 32)
dqn.main()