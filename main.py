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
from common.arguments import parse_args
from ActorCritic.A2C import A2C

class ActorCriticMain:
    def __init__(self, env, algo, feature_hidden_size, threads, log_interval, 
                 gamma, n_frames, n_steps, 
                 render_steps, tensorboardx_comment, target_reward,
                 model_save_path, evaluate_length, seed, **kwargs
                ):
        self.env = make_vec_envs(env, seed, threads, None, None, False)
        inputlen = self.env.observation_space.shape[0]
        self.GAMMA = gamma
        self.MAX_FRAMES = n_frames
        self.FRAME = 0
        self.model = cuda(ActorCriticNet(inputlen, feature_hidden_size, self.env.action_space.n, AtariCNN))
        algo = algo.lower()
        if algo == 'a2c':
            self.algo = A2C(self.model, threads, **kwargs)
        else:
            print('unknown algo', algo)
            raise NotImplementedError
        self.rollouts = cuda(RolloutStorage(n_steps, threads, self.env.observation_space.shape, 
                                            self.env.action_space, 1))
        self.threads = threads
        self.model.train()
        self.MAX_STEP = n_steps

        self.render = render_steps
        if tensorboardx_comment == '':
            self.TXSW = Fake_TXSW()
        else:
            self.TXSW = TXSW(comment = '_' + tensorboardx_comment)
        self.TXSW.add_graph(self.model, (cuda(torch.tensor(np.zeros((1, *self.env.observation_space.shape))).float()),))
        self.MODEL_SAVE_PATH = model_save_path
        self.TARGET_REWARD = target_reward
        self.PREVIOUS_REWARD = []
        self.BEST_RESULT = -1e100
        self.LOG_INTERVAL = log_interval
        self.EVAL_LENGTH = evaluate_length
    
    def main(self):
        last_time = time.time()
        last_step = 0
        state = self.env.reset()
        init_state = state
        self.rollouts.obs[0].copy_(torch.tensor(init_state))
        self.rollouts.step = 0
        tot_reward = torch.zeros(self.threads, 1).float()
        last_loss = 0
        find_terminate = False
        while self.FRAME < self.MAX_FRAMES:
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
                    pass
                    #pdb.set_trace()
                with torch.no_grad():
                    _, next_v = self.model(self.rollouts.obs[-1])
                    self.rollouts.compute_returns(next_v, False, self.GAMMA, 0, False)
                loss = self.algo.update_policy(self.rollouts, self.FRAME, self.TXSW)
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
                
            if self.FRAME % self.LOG_INTERVAL == 0 and len(self.PREVIOUS_REWARD) >= self.EVAL_LENGTH:
                print('Frame %7d, epoch %6d, %5d steps, %.1f steps/s, loss %4.4f, %4.2f'
                      % (self.FRAME, len(self.PREVIOUS_REWARD), self.FRAME * self.threads,
                         (self.FRAME * self.threads - last_step) / (time.time() - last_time), last_loss,
                         sum(self.PREVIOUS_REWARD[-self.EVAL_LENGTH:]) / self.EVAL_LENGTH
                        )
                     )
                last_time = time.time()
                last_step = self.FRAME * self.threads

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

if __name__ == "__main__":
    main = ActorCriticMain(**vars(parse_args()))
    main.main()