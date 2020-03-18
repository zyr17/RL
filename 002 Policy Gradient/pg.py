import torch
import numpy as np
import random
import pickle
#import mazeenv
import gym
import time
import collections
import cv2
import matplotlib.pyplot as plt
import pdb
import tensorboardX
TXSW = tensorboardX.SummaryWriter

import wrappers

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

class Fake_TXSW:
    def __init__(self):
        pass
    def add_scalar(self, *x):
        pass
    def add_image(self, *x):
        pass
    def add_graph(self, *x):
        pass
    def close(self):
        pass

def showarray(arr):
    arr = np.array(arr)
    print('max: %.2f, min: %.2f' % (arr.max(), arr.min()))
    plt.imshow(arr)
    plt.show()

#input: state; output: for every action: q values
class PGNet(torch.nn.Module):
    def __init__(self, inputlen, cnn, fc):
        super(PGNet, self).__init__()
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
        self.softmax = torch.nn.Softmax(dim = 1)

    def forward(self, inputs, apply_softmax = False):
        #print(inputs.shape)
        x = inputs
        for cnn in self.cnn:
            x = cnn(x)
            #print(x.shape)
        x = x.reshape(x.shape[0], -1)
        for fc in self.fc:
            x = fc(x)
            #print(x.shape)
        #pdb.set_trace()
        if apply_softmax:
            x = self.softmax(x)
        return x

class ReplayBuffer:
    def __init__(self, maxlen):
        self.position = 0
        self.buffer = []
        self.maxlen = maxlen

    def __len__(self):
        return len(self.buffer)

    def append(self, data):
        if len(self.buffer) < self.maxlen:
            self.buffer.append(data)
        else:
            self.buffer[self.position] = data
            self.position += 1

    def get(self, batch_size):
        choice = np.random.choice(self.maxlen, batch_size, replace = False)
        return [self.buffer[x] for x in choice]

class PG:
    def __init__(self, env, inputlen, cnn, fc, 
                 gamma = 0.95, epoch = 1000,
                 learning_rate = 0.001, batch_size = 128, reward_func = None,
                 max_step = 20, enable_advantage = True, baseline_length = 100, 
                 render = -1, TXComment = '', target_reward = 1e100,
                 model_save_path = ''
                ):
        self.env = env
        self.GAMMA = gamma
        self.EPOCH = epoch
        self.FRAME = 0
        self.model = cuda(PGNet(inputlen, cnn, fc))
        self.update_count = 0
        self.LR = learning_rate
        self.BATCH_SIZE = batch_size
        self.buffer = []
        self.opt = torch.optim.Adam(self.model.parameters(), learning_rate)
        self.model.train()
        self.reward_func = reward_func
        if type(reward_func) == type(None):
            self.reward_func = lambda env, state, reward: reward
        self.MAX_STEP = max_step
        self.ADVANTAGE = enable_advantage
        self.SAMPLE_REWARD = []
        self.BASELINE_LENGTH = baseline_length

        self.render = render
        if TXComment == '':
            self.TXSW = Fake_TXSW()
        else:
            self.TXSW = TXSW(comment = '_' + TXComment)
        self.TXSW.add_graph(self.model, (cuda(torch.tensor(np.zeros((self.BATCH_SIZE, *self.env.observation_space.shape))).float()),))
        self.MODEL_SAVE_PATH = model_save_path
        self.TARGET_REWARD = target_reward
        self.PREVIOUS_REWARD = []
        self.BEST_RESULT = -1e100

    def get_action(self, state):
        policy = self.model(cuda(torch.tensor([state]).float()), apply_softmax = True)[0].detach().cpu().numpy()
        return np.random.choice(len(policy), p = policy)

    def real_update_policy(self, state, action, reward, baseline):
        self.opt.zero_grad()
        state = cuda(state)
        action = cuda(action)
        reward = cuda(reward)
        if self.ADVANTAGE:
            reward -= baseline
        dist = self.model(state)
        log_s_dist = torch.nn.functional.log_softmax(dist)
        #pdb.set_trace()
        loss = -(reward * torch.gather(log_s_dist, 1, action.unsqueeze(1)).squeeze(1)).mean()
        loss.backward()
        self.opt.step()
        return loss.item()

    def update_policy(self, state, action, reward, baseline):
        self.buffer.append([state, action, reward])
        if len(self.buffer) == self.BATCH_SIZE:
            state_b, action_b, reward_b = zip(*self.buffer)
            state_b = torch.tensor(np.stack(state_b)).float()
            action_b = torch.tensor(np.stack(action_b))
            reward_b = torch.tensor(np.stack(reward_b))
            self.buffer.clear()
            return self.real_update_policy(state_b, action_b, reward_b, baseline)
        return None
    
    def sampling(self, epoch):
        start_time = time.time()
        state = self.env.reset()
        #init_state = state
        state_queue = collections.deque(maxlen = self.MAX_STEP)
        #state_queue.append([state, action, 0])
        #queue_reward = 0
        step = 0
        tot_reward = 0
        baseline = np.array(self.SAMPLE_REWARD[-self.BASELINE_LENGTH:]).mean() if len(self.SAMPLE_REWARD) > 0 else 0
        terminated = False
        while True:
            self.FRAME += 1
            step += 1
            if self.render != -1:
                self.env.render()
                time.sleep(self.render)
            
            if not terminated:
                action = self.get_action(state)
                next_s, reward, ist, _ = self.env.step(action)
                if ist: terminated = True
                reward = self.reward_func(self.env, next_s, reward)
                tot_reward += reward
                state_queue.append([state, action, reward])


            if len(state_queue) == self.MAX_STEP or terminated:
                rewards = 0
                for _, _, r in reversed(state_queue):
                    rewards = r + self.GAMMA * rewards
                self.SAMPLE_REWARD.append(rewards)
                loss = self.update_policy(state_queue[0][0], state_queue[0][1], torch.tensor(rewards).float(), baseline)
                if loss != None:
                    self.TXSW.add_scalar('loss', loss, self.FRAME - self.MAX_STEP)
                if terminated:
                    state_queue.popleft()
            
            if terminated and len(state_queue) == 0:
            #if ist:
                '''
                rewards = 0
                for _, _, r in reversed(state_queue):
                    rewards = r + self.GAMMA * rewards
                for i in range(1, len(state_queue)):
                    loss = self.update_policy(state_queue[i][0], state_queue[i][1], torch.tensor(rewards).float(), baseline)
                    if loss != None:
                        self.TXSW.add_scalar('loss', loss, self.FRAME - self.MAX_STEP + i)
                '''
                self.PREVIOUS_REWARD.append(tot_reward)
                self.TXSW.add_scalar('reward', tot_reward, epoch)
                print('Frame %7d, epoch %6d, %5d steps, %.1f steps/s, %4.2f, %4.2f' % (self.FRAME, epoch, step, step / (time.time() - start_time), tot_reward, baseline))
                break

            state = next_s
    
    def main(self):
        for ep in range(self.EPOCH):
            self.sampling(ep)
            if len(self.PREVIOUS_REWARD) > 10:
                now = sum(self.PREVIOUS_REWARD[-10:]) / 10
                if now > self.BEST_RESULT:
                    if self.BEST_RESULT != -1e100:
                        print('best result updated, %.4f -> %.4f.' % (self.BEST_RESULT, now))
                    self.BEST_RESULT = now
                    if self.MODEL_SAVE_PATH != '':
                        torch.save(self.model.state_dict(), self.MODEL_SAVE_PATH)
                    if now > self.TARGET_REWARD:
                        print('Problem solved, stop training.')
                        break
        self.TXSW.close()
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
    #return reward
    x = state[0]
    theta = state[2]
    r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.5
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    reward = float(r1 + r2)
    return reward
pg = PG(env, inputlen, cnn, fc, gamma = 0.99, learning_rate = 0.001, epoch = 100000, 
         batch_size = 16, max_step = 10, render = -1, target_reward = 200, baseline_length = 100, 
         reward_func=CartPole_reward_func, TXComment='PG_Cartpole')
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
fc = [7 * 7 * 64, 256, 6]
env = wrappers.make_env('PongNoFrameskip-v4')
pg = PG(env, inputlen, cnn, fc, gamma = 0.99, learning_rate = 0.0001, epoch = 100000, 
         batch_size = 32, max_step = 40, render = -1, target_reward = 18, baseline_length = 1000, 
         TXComment='PG_Pong')

pg.main()
