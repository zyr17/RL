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

from common.envs import TransposeImage, make_vec_envs
from common.models import AtariCNN, ActorCriticNet
from common.utils import cuda, Fake_TXSW
from common.storage import RolloutStorage

class A2C:
    def __init__(self, envs, hiddenlen = 512, threads = 1, log_interval = 100, 
                 gamma = 0.95, max_frames = 1000000, num_envs = 10,
                 learning_rate = 0.001, batch_size = 128, 
                 max_step = 20, entropy_beta = 0.01, clip_grad = 0.1,
                 render = -1, TXComment = '', target_reward = 1e100,
                 model_save_path = '', evaluate_length = 10
                ):
        inputlen = envs.observation_space.shape[0]
        self.env = envs
        self.GAMMA = gamma
        self.MAX_FRAMES = max_frames
        self.FRAME = 0
        self.model = cuda(ActorCriticNet(inputlen, hiddenlen, envs.action_space.n, AtariCNN))
        self.update_count = 0
        self.LR = learning_rate
        self.BATCH_SIZE = batch_size
        self.rollouts = cuda(RolloutStorage(max_step, threads, envs.observation_space.shape, 
                                            envs.action_space, 1))
        self.threads = threads
        self.opt = torch.optim.Adam(self.model.parameters(), learning_rate)
        self.model.train()
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
        self.LOG_INTERVAL = log_interval
        self.EVAL_LENGTH = evaluate_length

    def real_update_policy(self):
        #pdb.set_trace()
        #print('reward sum', sum(reward))
        #state, _, action, _, next_v, reward, ist, _, next_state, returns = self.rollouts.get_flatten()
        state, action, reward, next_state, returns = self.rollouts.get_flatten('obs', 'actions', 'rewards', self.rollouts.obs[1:], 'returns')
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
        '''
        if not hasattr(self, 'LAST_OUTPUT') or self.FRAME - 10000 > self.LAST_OUTPUT:
            print(pd_loss.item(), self.ENTROPY_BETA * entropy_loss.item(), v_loss.item(), v.mean().item())
            self.LAST_OUTPUT = self.FRAME
        '''
        self.TXSW.add_scalar('pd_loss', pd_loss.item(), self.FRAME)
        self.TXSW.add_scalar('entropy_loss', entropy_loss.item(), self.FRAME)
        self.TXSW.add_scalar('v_loss', v_loss.item(), self.FRAME)
        self.TXSW.add_scalar('v_mean', v.mean().item(), self.FRAME)
        loss = pd_loss + self.ENTROPY_BETA * entropy_loss + v_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.CLIP_GRAD)
        self.opt.step()
        return loss.item()
    
    def main(self):
        start_time = time.time()
        state = self.env.reset()
        init_state = state
        self.rollouts.obs[0].copy_(torch.tensor(init_state))
        self.rollouts.step = 0
        tot_reward = torch.zeros(self.threads, 1).float()
        last_loss = 0
        find_terminate = False
        while True:
            self.FRAME += 1
            if self.render > 0 and self.FRAME % self.render == 0:
                self.env.render()
            #pdb.set_trace()
            
            action = self.model.get_action(state)
            next_s, reward, ist, info = self.env.step(action)
            tot_reward += reward
            self.rollouts.insert(next_s, np.zeros((self.threads, 1)), action, np.ones((self.threads, 1)),
                                 np.zeros((self.threads, 1)), reward, np.expand_dims(1 - ist, 1), np.zeros((self.threads, 1)))

            if self.rollouts.step == self.MAX_STEP:
                if find_terminate:
                    pass#pdb.set_trace()
                with torch.no_grad():
                    _, next_v = self.model(self.rollouts.obs[-1])
                    self.rollouts.compute_returns(next_v, False, self.GAMMA, 0, False)
                loss = self.real_update_policy()
                self.rollouts.after_update()
                self.TXSW.add_scalar('loss', loss, self.FRAME - self.MAX_STEP)
                last_loss = loss
                find_terminate = False

            for num, i in enumerate(ist):
                if i:
                    # if multilive, don't push to self.PREVIOUS_REWARD. TODO: record every live seperately
                    if 'ale.lives' in info[num].keys() and info[num]['ale.lives'] > 0:
                        continue
                    find_terminate = True
                    self.PREVIOUS_REWARD.append(tot_reward[num].item())
                    self.TXSW.add_scalar('reward', tot_reward[num], len(self.PREVIOUS_REWARD))
                    tot_reward[num] = 0
                    if self.evaluate_terminate(self.EVAL_LENGTH):
                        return
                
            if self.FRAME % self.LOG_INTERVAL == 0 and len(self.PREVIOUS_REWARD) > 0:
                print('Frame %7d, epoch %6d, %5d steps, %.1f steps/s, loss %4.4f, %4.2f'
                      % (self.FRAME, len(self.PREVIOUS_REWARD), self.FRAME * self.threads,
                         self.FRAME * self.threads / (time.time() - start_time), last_loss,
                         sum(self.PREVIOUS_REWARD[-self.EVAL_LENGTH:]) / self.EVAL_LENGTH
                        )
                     )

            state = next_s

    def evaluate_terminate(self, evaluate_length = 10):
        if len(self.PREVIOUS_REWARD) > evaluate_length:
            now = sum(self.PREVIOUS_REWARD[-evaluate_length:]) / evaluate_length
            if now > self.BEST_RESULT:
                if self.BEST_RESULT != -1e100:
                    print('best result updated, %.4f -> %.4f.' % (self.BEST_RESULT, now))
                self.BEST_RESULT = now
                if self.MODEL_SAVE_PATH != '':
                    torch.save(self.model.state_dict(), self.MODEL_SAVE_PATH)
                if now > self.TARGET_REWARD:
                    print('Problem solved, stop training.')
                    self.TXSW.close()
                    return True
        return False
            
def a2c_main():
    #Pong CNN
    threads = 32
    envs = make_vec_envs('BreakoutNoFrameskip-v4', 0, threads, None, None, False)
    a2c = A2C(envs, gamma = 0.99, learning_rate = 0.0001, max_frames = 10000000, threads = threads, 
            batch_size = 64, max_step = 5, render = -1, target_reward = 300,
            TXComment='')
    a2c.main()

if __name__ == "__main__":
    a2c_main()