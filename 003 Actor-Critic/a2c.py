import torch
import numpy as np
import random
import pickle
import gym
import time
import collections
import cv2
import matplotlib.pyplot as plt
import pdb
import tensorboardX
from baselines.common.atari_wrappers import wrap_deepmind, make_atari
TXSW = tensorboardX.SummaryWriter

import sys
sys.path.append("..")

from common.wrappers import ChangeAxis
from common.models import AtariCNN
from common.utils import cuda, Fake_TXSW
from common.storage import RolloutStorage

try:
    @profile
    def emptyfunc():
        pass
except:
    def profile(func):
        return func

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
    def __init__(self, env, inputlen, hiddenlen = 512, 
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
        self.model = cuda(A2CNet(inputlen, hiddenlen, env.action_space.n, AtariCNN))
        self.update_count = 0
        self.LR = learning_rate
        self.BATCH_SIZE = batch_size
        self.rollouts = cuda(RolloutStorage(max_step, 1, env.observation_space.shape, 
                                            env.action_space, 1))
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
        state = cuda(torch.tensor(np.expand_dims(state, 0)).float())
        #state = cuda(torch.tensor([state]).float())
        with torch.no_grad():
            policy = self.model(state, apply_softmax = True)[0][0]
        policy = policy.detach().cpu().numpy()
        return np.random.choice(len(policy), p = policy)

    def real_update_policy(self):
        #pdb.set_trace()
        #print('reward sum', sum(reward))
        state, _, action, _, next_v, reward, ist, _, next_state, returns = self.rollouts.get_flatten()
        self.opt.zero_grad()
        state = cuda(state)
        action = cuda(action)
        reward = cuda(returns)
        next_state = cuda(next_state)
        dist, v = self.model(state)
        log_s_dist = torch.nn.functional.log_softmax(dist, dim = 1)
        s_dist = torch.nn.functional.softmax(dist, dim = 1)
        #pdb.set_trace()
        pd_loss = -((reward - v.detach()) * torch.gather(log_s_dist, 1, action)).mean()
        entropy_loss = (s_dist * log_s_dist).sum(dim=1).mean()
        #print(next_v + reward.unsqueeze(1), v)
        #pdb.set_trace()
        v_loss = torch.nn.functional.mse_loss(reward, v)

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
        init_state = state
        self.rollouts.obs[0].copy_(torch.tensor(np.expand_dims(init_state, 0)))
        step = 0
        self.rollouts.step = 0
        tot_reward = 0
        terminated = False
        last_loss = 0
        while True:
            self.FRAME += 1
            step += 1
            if self.render > 0 and self.FRAME % self.render == 0:
                self.env.render()
            #pdb.set_trace()
            if not terminated:
                action = self.get_action(state)
                next_s, reward, ist, _ = self.env.step(action)
                if ist:
                    terminated = True
                reward = self.reward_func(self.env, next_s, reward)
                tot_reward += reward
                self.rollouts.insert(np.expand_dims(next_s, 0), np.zeros((1, 1)), np.expand_dims(np.array(action), 0), np.ones((1, 1)),
                                     np.zeros((1, 1)), np.expand_dims(reward, 0), np.expand_dims(np.array(not ist), 0), np.zeros((1, 1)))

            if self.rollouts.step == self.MAX_STEP or terminated:
                with torch.no_grad():
                    _, next_v = self.model(self.rollouts.obs[-1])
                    self.rollouts.compute_returns(next_v, False, self.GAMMA, 0, False)
                loss = self.real_update_policy()
                self.rollouts.after_update()
                if loss != None:
                    self.TXSW.add_scalar('loss', loss, self.FRAME - self.MAX_STEP)
                    last_loss = loss
            
            if terminated:
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

#Pong CNN
inputlen = 1
env = ChangeAxis(wrap_deepmind(make_atari('PongNoFrameskip-v4')))
a2c = A2C(env, inputlen, gamma = 0.99, learning_rate = 0.0001, epoch = 100000, 
         batch_size = 64, max_step = 40, render = -1, target_reward = 18,
         TXComment='A2C_Pong')

a2c.main()
