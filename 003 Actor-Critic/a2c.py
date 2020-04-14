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

import sys
sys.path.append("..")

from common.wrappers import ChangeAxis
from common.models import AtariCNN

from baselines.common.atari_wrappers import wrap_deepmind, make_atari

try:
    @profile
    def emptyfunc():
        pass
except:
    def profile(func):
        return func

ENABLE_CUDA = True
#torch.backends.cudnn.benchmark = True

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

#input: state; output: policy distribution, value function
class A2CNet(torch.nn.Module):
    def __init__(self, inputlen, featurelen, action_space, feature_net):
        super(A2CNet, self).__init__()
        self.feature_net = feature_net(inputlen, featurelen)
        self.p_fc = torch.nn.Linear(featurelen, action_space)
        self.softmax = torch.nn.Softmax(dim = 1)
        self.v_fc = torch.nn.Linear(featurelen, 1)

    def forward(self, inputs, apply_softmax = False):
        x = inputs
        x = self.feature_net(x)
        v = self.v_fc(x)
        pg = self.p_fc(x)
        if apply_softmax:
            pg = self.softmax(pg)
        return pg, v

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

class A2C:
    def __init__(self, env, inputlen, cnn, fc, 
                 gamma = 0.95, epoch = 1000,
                 learning_rate = 0.001, batch_size = 128, reward_func = None,
                 max_step = 20, entropy_beta = 0.01, clip_grad = 0.1,
                 render = -1, TXComment = '', target_reward = 1e100,
                 model_save_path = ''
                ):
        self.env = env
        self.GAMMA = gamma
        self.EPOCH = epoch
        self.FRAME = 0
        self.model = cuda(A2CNet(inputlen, 512, env.action_space.n, AtariCNN))
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
        self.ENTROPY_BETA = entropy_beta
        self.CLIP_GRAD = clip_grad
        self.SAMPLE_REWARD = []

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

    @profile
    def get_action(self, state):
        #pdb.set_trace()
        state = np.expand_dims(state, 0)
        state = torch.tensor(state)
        state = state.float()
        state = cuda(state)
        #state = cuda(torch.tensor([state]).float())
        with torch.no_grad():
            policy = self.model(state, apply_softmax = True)[0][0]
        policy = policy.detach().cpu().numpy()
        return np.random.choice(len(policy), p = policy)

    def real_update_policy(self, state, action, reward, next_state, ist):
        #pdb.set_trace()
        #print('reward sum', sum(reward))
        self.opt.zero_grad()
        state = cuda(state)
        action = cuda(action)
        reward = cuda(reward)
        next_state = cuda(next_state)
        dist, v = self.model(state)
        _, next_v = self.model(next_state)
        next_v.detach_()
        next_v[ist] = 0
        next_v *= self.GAMMA ** self.MAX_STEP
        next_v.squeeze_()
        v.squeeze_()
        #print(next_v.shape, reward.shape)
        next_v += reward
        log_s_dist = torch.nn.functional.log_softmax(dist, dim = 1)
        s_dist = torch.nn.functional.softmax(dist, dim = 1)
        #pdb.set_trace()
        pd_loss = -((next_v - v.detach()) * torch.gather(log_s_dist, 1, action.unsqueeze(1)).squeeze(1)).mean()
        entropy_loss = (s_dist * log_s_dist).sum(dim=1).mean()
        #print(next_v + reward.unsqueeze(1), v)
        #pdb.set_trace()
        v_loss = torch.nn.functional.mse_loss(next_v, v)

        if not hasattr(self, 'LAST_OUTPUT') or self.FRAME - 10000 > self.LAST_OUTPUT:
            print(pd_loss.item(), self.ENTROPY_BETA * entropy_loss.item(), v_loss.item(), v.mean().item())
            self.LAST_OUTPUT = self.FRAME
        loss = pd_loss + self.ENTROPY_BETA * entropy_loss + v_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.CLIP_GRAD)
        self.opt.step()
        return loss.item()

    def update_policy(self, state, action, reward, next_state, ist):
        self.buffer.append([state, action, reward, next_state, ist])
        if len(self.buffer) == self.BATCH_SIZE:
            state_b, action_b, reward_b, next_state_b, ist_b = zip(*self.buffer)
            state_b = torch.tensor(np.stack(state_b)).float()
            action_b = torch.tensor(np.stack(action_b))
            reward_b = torch.tensor(np.stack(reward_b))
            next_state_b = torch.tensor(np.stack(next_state_b)).float()
            ist_b = torch.tensor(np.stack(ist_b))
            self.buffer.clear()
            return self.real_update_policy(state_b, action_b, reward_b, next_state_b, ist_b)
        return None
    
    @profile
    def sampling(self, epoch):
        start_time = time.time()
        state = self.env.reset()
        #init_state = state
        state_queue = collections.deque(maxlen = self.MAX_STEP)
        #state_queue.append([state, action, 0])
        #queue_reward = 0
        step = 0
        tot_reward = 0
        terminated = False
        last_loss = 0
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
                loss = self.update_policy(state_queue[0][0], state_queue[0][1], torch.tensor(rewards).float(), next_s, terminated)
                if loss != None:
                    self.TXSW.add_scalar('loss', loss, self.FRAME - self.MAX_STEP)
                    last_loss = loss
                if terminated:
                    state_queue.popleft()
            
            if terminated and len(state_queue) == 0:
                self.PREVIOUS_REWARD.append(tot_reward)
                self.TXSW.add_scalar('reward', tot_reward, epoch)
                print('Frame %7d, epoch %6d, %5d steps, %.1f steps/s, loss %4.4f, %4.2f' % (self.FRAME, epoch, step, step / (time.time() - start_time), last_loss, tot_reward))
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
fc = [4, 100, 10]
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
a2c = A2C(env, inputlen, cnn, fc, gamma = 0.99, learning_rate = 0.001, epoch = 100000, 
         batch_size = 16, max_step = 10, render = -1, target_reward = 200,
         reward_func=CartPole_reward_func, TXComment='A2C_Cartpole')
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
inputlen = 1
cnn = [
    (32, 8, 0, 4, 1, 0),
    (64, 4, 0, 2, 1, 0),
    (64, 3, 0, 1, 1, 0),
]
fc = [7 * 7 * 64, 256]
env = ChangeAxis(wrap_deepmind(make_atari('PongNoFrameskip-v4')))
a2c = A2C(env, inputlen, cnn, fc, gamma = 0.99, learning_rate = 0.0001, epoch = 100000, 
         batch_size = 64, max_step = 40, render = -1, target_reward = 18,
         TXComment='A2C_Pong')

a2c.main()
