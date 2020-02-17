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
        return x

class replay_array:
    def __init__(self, replay_length, priority = False):
        self.REPLAY = replay_length
        self.replay_data = collections.deque(maxlen = replay_length)
        self.priority = np.zeros((replay_length * 3), 'float')
        self.priority_default = 1.0
        self.priority_update = 0
        self.priority_position = 0

    def __len__(self):
        return len(self.replay_data)

    def append(self, data):
        self.replay_data.append(data)
        self.priority[self.priority_position] = self.priority_default
        self.priority_position += 1
        self.priority_update += 1
        if self.priority_update == self.REPLAY:
            #reset self.priority_default
            self.priority_default = self.priority[self.priority_position - self.REPLAY:self.priority_position].max()
            self.priority_update = 0
        if self.priority_position > self.REPLAY * 2 + 2:
            self.priority[:self.REPLAY] = self.priority[self.priority_position - self.REPLAY:self.priority_position]
            self.priority_position = self.REPLAY

    def sample(self, size, beta = 0.4):

        prob = self.priority[self.priority_position - self.REPLAY:self.priority_position]
        '''
        print([[x[1], x[2].item(), x[4]] for x in self.replay_data])
        print(self.priority[self.priority_position - self.REPLAY:self.priority_position])
        input()
        '''
        choice = np.random.choice(self.REPLAY, size, replace = False, p = prob / prob.sum())
        weight = (prob[choice] * self.REPLAY) ** (-beta)
        weight /= weight.max()
        return choice, zip(*[self.replay_data[x] for x in choice]), weight

    def change_weight(self, idx, weight):
        weight = np.abs(weight)
        for id, w in zip(idx, weight):
            self.priority[self.priority_position - self.REPLAY + id] = w
            if w > self.priority_default:
                self.priority_default = w
                self.priority_update = 0

class DQN:
    def __init__(self, env, inputlen, cnn, fc, 
                 alpha = 0.1, gamma = 0.95, eps = 0.1, 
                 epoch = 1000, replay = 1000000, update_round = 10000,
                 learning_rate = 0.001, batch_size = 128, reward_func = None,
                 render = -1, double = False, TXComment = '', target_reward = 1e100,
                 model_save_path = '', prioritized = False
                ):
        self.env = env
        self.ALPHA = alpha
        self.BETA = 0.4
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
        self.model_old = cuda(DQNnet(inputlen, cnn, fc))
        self.model_update = cuda(DQNnet(inputlen, cnn, fc))
        self.model_old.load_state_dict(self.model_update.state_dict())
        self.update_count = 0
        self.replay_data = replay_array(replay)
        self.LR = learning_rate
        self.BATCH_SIZE = batch_size
        self.opt = torch.optim.Adam(self.model_update.parameters(), learning_rate)
        def lossf(a, b):
            res = (a - b) * (a - b)
            return res.mean(1)
        self.PRIORITIZED = prioritized
        self.loss = lossf
        self.model_old.eval()
        self.model_update.train()
        self.reward_func = reward_func
        if type(reward_func) == type(None):
            self.reward_func = lambda env, state, reward: reward
        self.render = render
        self.DOUBLE = double
        self.TXSW = TXSW(comment = '_' + TXComment)
        self.TXSW.add_graph(self.model_old, (cuda(torch.tensor(np.zeros((self.BATCH_SIZE, *self.env.observation_space.shape))).float()),))
        self.MODEL_SAVE_PATH = model_save_path
        self.TARGET_REWARD = target_reward
        self.PREVIOUS_REWARD = []
        self.BEST_RESULT = -1e100
        
        self.LOSS_WRITE_COUNT = 0

    def get_action(self, state):
        if random.random() < self.EPS:
            return self.env.action_space.sample()
        q = self.model_update(cuda(torch.tensor([state]).float()))[0]
        if False and self.REPLAY == len(self.replay_data):
            print(q, ' choice:', torch.argmax(q).item(), ' ', end = '')
            showarray(state[-1])
        return torch.argmax(q).item()

    def real_update_q(self, state, action, reward, next_s, ist, weight):
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
        weight = cuda(torch.tensor(weight).float())
        q = self.model_update(state)
        #print(q)
        action = action.reshape(action.shape[0], 1)
        if self.DOUBLE:
            next_a = self.model_update(next_s).max(dim = 1)[1]
            next_q = self.model_old(next_s).gather(1, next_a.unsqueeze(1)).squeeze(1)
        else:
            next_q = self.model_old(next_s).max(dim = 1)[0]
        reward_b = reward + self.GAMMA * next_q * cuda(torch.tensor(1 - ist).float())
        reward_b = reward_b.reshape(reward_b.shape[0], 1)
        #reward_b = reward_b.detach()
        L = self.loss(q.gather(1, action), reward_b)
        for i in L:
            self.LOSS_WRITE_COUNT += 1
            self.TXSW.add_scalar('loss_original', i, self.LOSS_WRITE_COUNT)
        L = L * weight
        LL = L.detach().cpu().numpy()
        L = L.mean()
        L.backward()
        self.opt.step()
        return L.item(), LL + 1e-5

    def update_q(self, state, action, reward, next_s, ist):
        self.replay_data.append([state, action, reward, next_s, ist])
        if len(self.replay_data) == self.REPLAY:
            idx, data, weight = self.replay_data.sample(self.BATCH_SIZE, self.BETA)
            loss, one_loss = self.real_update_q(*data, weight)
            if self.PRIORITIZED:
                self.replay_data.change_weight(idx, one_loss)
            self.update_count = (self.update_count + 1) % self.UPDATE
            if self.update_count == 0:
                self.model_old.load_state_dict(self.model_update.state_dict())
            return loss
        return None
    
    def sampling(self, epoch):
        start_time = time.time()
        state = self.env.reset()
        init_state = state
        action = self.get_action(state)
        step = 0
        tot_reward = 0
        while True:
            self.FRAME += 1
            self.EPS = max(self.EPS - self.EPS_STEP, self.EPS_MIN)
            self.BETA = min(1, self.BETA + self.EPS_STEP)
            step += 1
            if self.render != -1:
                self.env.render()
                time.sleep(self.render)
            next_s, reward, ist, _ = self.env.step(action)
            
            #pickle.dump(next_s, open('input/%06d.pt' % self.FRAME, 'wb'))
            #if self.FRAME == 1000: exit()

            reward = self.reward_func(self.env, next_s, reward)
            tot_reward += reward
            loss = self.update_q(state, action, torch.tensor(reward).float(), next_s, 1 if ist else 0)
            if loss != None:
                self.TXSW.add_scalar('loss', loss, self.FRAME)
            if len(self.replay_data) == self.REPLAY and self.update_count == 0:
                img = np.zeros((1, 84, 84 // 6 * 7), dtype='float')
                img[0,:,:84] = state[-1]
                now_act = self.model_old(cuda(torch.tensor(state).float().unsqueeze(0))).cpu()[0]
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
            action = self.get_action(state)
    
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
fc = [7 * 7 * 64, 256, 6]
env = wrappers.make_env('PongNoFrameskip-v4')

dqn = DQN(env, inputlen, cnn, fc, gamma = 0.99, learning_rate = 0.0001, eps = [1, 0.00001, 0.02],
          epoch = 100000, replay = 10000, update_round = 1000, render = -1, batch_size = 32, prioritized = True,
          TXComment = 'Prioritized', target_reward = 19.5, model_save_path = 'models/Prioritized.pt')

'''
dqn = DQN(env, inputlen, cnn, fc, gamma = 0.99, learning_rate = 0.0001, eps = [1, 0.00001, 0.02],
          epoch = 100000, replay = 10000, update_round = 1000, render = -1, batch_size = 32, double = True,
          TXComment = 'Double_DQN', target_reward = 19.5, model_save_path = 'models/Double_DQN.pt')
'''
dqn.main()