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
import pdb
import tensorboardX
TXSW = tensorboardX.SummaryWriter

class Fake_TXSW:
    def __init__(self):
        pass
    def add_scalar(self, *x):
        pass
    def add_image(self, *x):
        pass
    def add_graph(self, *x):
        pass

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

def showarray(arr):
    arr = np.array(arr)
    print('max: %.2f, min: %.2f' % (arr.max(), arr.min()))
    plt.imshow(arr)
    plt.show()

#input: state; output: for every action: q values
class DQNnet(torch.nn.Module):
    def __init__(self, inputlen, cnn, fc, n_atoms = 51, value_min = 0, value_max = 1):
        super(DQNnet, self).__init__()
        self.const = {}
        self.const['inputlen'] = inputlen
        self.const['cnn'] = cnn
        self.const['fc'] = fc
        self.cnn = torch.nn.ModuleList()
        self.n_atoms = n_atoms
        self.value_min = value_min
        self.value_max = value_max
        self.register_buffer("value_arr", torch.tensor(range(n_atoms)).float() / (n_atoms - 1) * (value_max - value_min) + value_min)
        #print(self.value_arr)
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
        self.softmax = torch.nn.Softmax(dim = 2)

    def forward(self, inputs):
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
        #print(x.shape)
        x = x.reshape(x.shape[0], -1, self.n_atoms)
        #print(x.shape)
        return x

    def apply_softmax(self, x):
        return self.softmax(x)

    def expectation(self, x):
        #print('exp', x.shape)
        probs = self.apply_softmax(x)
        #print(probs.shape)
        value = probs * self.value_arr
        #print(res.shape)
        res = value.sum(dim = 2)
        #print(res.shape)
        #pdb.set_trace()
        return res

class DQN:
    def __init__(self, env, inputlen, cnn, fc, 
                 alpha = 0.1, gamma = 0.95, eps = 0.1, 
                 epoch = 1000, replay = 1000000, update_round = 10000,
                 learning_rate = 0.001, batch_size = 128, reward_func = None,
                 render = -1, double = False, TXComment = '', target_reward = 1e100,
                 model_save_path = '', n_step = 1, n_atoms = 1, value_min = 0, value_max = 1
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
        self.FRAME = 0
        self.REPLAY = replay
        self.UPDATE = update_round
        self.model_old = cuda(DQNnet(inputlen, cnn, fc, n_atoms, value_min, value_max))
        self.model_update = cuda(DQNnet(inputlen, cnn, fc, n_atoms, value_min, value_max))
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
        self.DOUBLE = double
        self.N_STEP = n_step
        self.N_ATOMS = n_atoms
        self.VALMAX = value_max
        self.VALMIN = value_min
        self.TXSW = TXSW(comment = '_' + TXComment) if TXComment != '' else Fake_TXSW()
        self.TXSW.add_graph(self.model_old, (cuda(torch.tensor(np.zeros((self.BATCH_SIZE, *self.env.observation_space.shape))).float()),))
        self.MODEL_SAVE_PATH = model_save_path
        self.TARGET_REWARD = target_reward
        self.PREVIOUS_REWARD = []
        self.BEST_RESULT = -1e100

    def get_action(self, state):
        if random.random() < self.EPS:
            return self.env.action_space.sample()
        q = self.model_update.expectation(self.model_update(cuda(torch.tensor([state]).float())))[0]
        if False and self.REPLAY == len(self.replay_data):
            print(q, ' choice:', torch.argmax(q).item(), ' ', end = '')
            showarray(state[-1])
        return torch.argmax(q).item()

    def dist_proj(self, next_dist, reward, ist):
        #pdb.set_trace()
        res_dist = torch.zeros(next_dist.shape).float()
        #res_dist = cuda(res_dist)
        delta = (self.VALMAX - self.VALMIN) / (self.N_ATOMS - 1)
        for atom in range(self.N_ATOMS):
            r = reward + self.GAMMA * (self.VALMIN + atom * delta)
            r[r > self.VALMAX] = self.VALMAX
            r[r < self.VALMIN] = self.VALMIN
            idx = (r - self.VALMIN) / delta
            low = torch.floor(idx).long()
            high = torch.ceil(idx).long()
            eq_msk = low == high
            ne_msk = low != high
            res_dist[eq_msk, low[eq_msk]] += next_dist[eq_msk, atom]
            res_dist[ne_msk, low[ne_msk]] += next_dist[ne_msk, atom] * (high - idx)[ne_msk]
            res_dist[ne_msk, high[ne_msk]] += next_dist[ne_msk, atom] * (idx - low)[ne_msk]
        
        #pdb.set_trace()
        if ist.sum() > 0:
            #pdb.set_trace()
            ist_msk = torch.tensor(ist).bool()
            res_dist[ist_msk] = 0.0
            r = reward
            r[r > self.VALMAX] = self.VALMAX
            r[r < self.VALMIN] = self.VALMIN
            idx = (r - self.VALMIN) / delta
            low = torch.floor(idx).long()
            high = torch.ceil(idx).long()
            #ist_msk = cuda(ist_msk)
            eq_msk = (low == high) * ist_msk
            ne_msk = (low != high) * ist_msk
            res_dist[eq_msk, low[eq_msk]] = 1.0
            res_dist[ne_msk, low[ne_msk]] = (high - idx)[ne_msk]
            res_dist[ne_msk, high[ne_msk]] = (idx - low)[ne_msk]
            #pdb.set_trace()

        #pdb.set_trace()
        return res_dist

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
        reward_cpu = torch.tensor(reward).float()
        reward = cuda(reward_cpu)
        next_s = cuda(torch.tensor(next_s).float())
        dist = self.model_update(state)
        q = self.model_update.expectation(dist)
        #print(q)
        action = action.reshape(action.shape[0], 1, 1).repeat(1, 1, self.N_ATOMS)
        if self.DOUBLE:
            next_a = self.model_update.expectation(self.model_update(next_s)).max(dim = 1)[1]
            next_dist = self.model_old.apply_softmax(self.model_old(next_s)).gather(1, next_a.unsqueeze(1).unsqueeze(2).repeat(1, 1, self.N_ATOMS)).squeeze(1)
            next_q = self.model_old.expectation(next_dist)
        else:
            next_dists = self.model_old(next_s)
            next_q, idx = self.model_old.expectation(next_dists).max(dim = 1)
            next_dist = self.model_old.apply_softmax(next_dists).gather(1, idx.unsqueeze(1).unsqueeze(2).repeat(1, 1, self.N_ATOMS)).squeeze(1)
        #next_dist: [BATCH, N_ATOMS]
        #pdb.set_trace()
        
        new_dist = cuda(self.dist_proj(next_dist.cpu(), reward_cpu, ist))
        #pdb.set_trace()
        L = (-new_dist * torch.nn.functional.log_softmax(dist.gather(1, action).squeeze(1), dim = 1)).sum(dim = 1).mean()
        
        #reward_b = reward + (self.GAMMA ** self.N_STEP) * next_q * cuda(torch.tensor(1 - ist).float())
        #reward_b = reward_b.reshape(reward_b.shape[0], 1)
        #reward_b = reward_b.detach()
        #L = self.loss(q.gather(1, action), reward_b)
        L.backward()
        self.opt.step()
        return L.item()

    def update_q(self, state, action, reward, next_s, ist):
        self.replay_data.append([state, action, reward, next_s, ist])
        if len(self.replay_data) == self.REPLAY:
            choice = np.random.choice(self.REPLAY, self.BATCH_SIZE, replace = False)
            loss = self.real_update_q(*zip(*[self.replay_data[x] for x in choice]))
            self.update_count = (self.update_count + 1) % self.UPDATE
            if self.update_count == 0:
                self.model_old.load_state_dict(self.model_update.state_dict())
            return loss
        return None
    
    def sampling(self, epoch):
        start_time = time.time()
        state = self.env.reset()
        #init_state = state
        state_queue = collections.deque(maxlen = self.N_STEP + 1)
        action = self.get_action(state)
        state_queue.append([state, action, 0])
        step = 0
        tot_reward = 0
        while True:
            self.FRAME += 1
            self.EPS = max(self.EPS - self.EPS_STEP, self.EPS_MIN)
            step += 1
            if self.render != -1:
                self.env.render()
                time.sleep(self.render)
            next_s, reward, ist, _ = self.env.step(action)
            reward = self.reward_func(self.env, next_s, reward)
            next_action = self.get_action(next_s)
            state_queue.append([next_s, next_action, reward])
            if len(state_queue) == self.N_STEP + 1:
                
                #pickle.dump(next_s, open('input/%06d.pt' % self.FRAME, 'wb'))
                #if self.FRAME == 1000: exit()

                rewards = 0
                for s, a, r in state_queue:
                    rewards += r
                tot_reward += reward
                loss = self.update_q(state_queue[0][0], state_queue[0][1], torch.tensor(rewards).float(), state_queue[-1][0], 1 if ist else 0)
                if loss != None:
                    self.TXSW.add_scalar('loss', loss, self.FRAME)
                if len(self.replay_data) == self.REPLAY and self.update_count == 0:
                    img = np.zeros((1, 84, 84 // 6 * 7), dtype='float')
                    img[0,:,:84] = state[-1]
                    now_act = self.model_old.expectation(self.model_old(cuda(torch.tensor(state).float().unsqueeze(0)))).cpu()[0]
                    now_act -= now_act.min()
                    if now_act.max() != 0:
                        now_act /= now_act.max()
                    #print(now_act)
                    #now_act = torch.nn.Softmax(dim=0)(now_act)
                    now_act = now_act.detach().numpy()
                    #print(now_act)
                    for i in range(6):
                        img[0,i*14:i*14+14,84:] = now_act[i]# * 256
                    #pdb.set_trace()
                    self.TXSW.add_image('model_old', img, self.FRAME)
            
            if ist:
                self.PREVIOUS_REWARD.append(tot_reward)
                self.TXSW.add_scalar('reward', tot_reward, epoch)
                print('Frame %9d, epoch %6d, %6d steps, %.2f steps/s, eps %.2f' % (self.FRAME, epoch, step, step / (time.time() - start_time), self.EPS), tot_reward)
                break

            state = next_s
            action = next_action
    
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
                        torch.save(self.model_old.state_dict(), self.MODEL_SAVE_PATH)
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
n_atoms = 51
fc = [7 * 7 * 64, 1000, 6 * n_atoms]
env = wrappers.make_env('PongNoFrameskip-v4')

dqn = DQN(env, inputlen, cnn, fc, gamma = 0.99, learning_rate = 0.0001, eps = [1, 0.00001, 0.02],
          epoch = 100000, replay = 10000, update_round = 1000, render = -1, batch_size = 32, n_atoms = n_atoms, value_min = -21, value_max = 21,
          TXComment = '', target_reward = 19.5, model_save_path = '')
'''
dqn = DQN(env, inputlen, cnn, fc, gamma = 0.99, learning_rate = 0.0001, eps = [1, 0.00001, 0.02],
          epoch = 100000, replay = 10000, update_round = 1000, render = -1, batch_size = 32, double = True,
          TXComment = 'Double_DQN', target_reward = 19.5, model_save_path = 'models/Double_DQN.pt')

dqn = DQN(env, inputlen, cnn, fc, gamma = 0.99, learning_rate = 0.0001, eps = [1, 0.00001, 0.02],
          epoch = 100000, replay = 10000, update_round = 1000, render = -1, batch_size = 32, n_step = 5,
          TXComment = '5_Step_DQN', target_reward = 19.5, model_save_path = 'models/5_Step_DQN.pt')
'''
dqn.main()