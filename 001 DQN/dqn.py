import torch
import numpy as np
import random
import pickle
import mazeenv
import gym

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
                torch.nn.Conv2d(lastfea, i[0], i[1], padding=i[2]),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(i[3], padding=i[4])
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
                 alpha = 0.1, gamma = 0.95, eps = 0.9, 
                 epoch = 1000, replay = 1000000, update_round = 10000,
                 learning_rate = 0.001, batch_size = 128, reward_func = None
                ):
        self.env = env
        self.ALPHA = alpha
        self.GAMMA = gamma
        self.EPS = eps
        self.EPOCH = epoch
        self.REPLAY = replay
        self.UPDATE = update_round
        self.model_old = DQNnet(inputlen, cnn, fc)
        self.model_update = DQNnet(inputlen, cnn, fc)
        self.model_old.load_state_dict(self.model_update.state_dict())
        self.update_count = 0
        self.replay_count = 0
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

    def get_action(self, state):
        if random.random() > self.EPS:
            return self.env.action_space.sample()
        q = self.model_update(torch.tensor([state]).float())[0]
        return torch.argmax(q).item()

    def real_update_q(self, state, action, reward, next_s, ist):
        #print('real update q', state, action, reward)
        self.opt.zero_grad()
        state = torch.tensor(state).float()
        next_s = torch.tensor(next_s).float()
        q = self.model_update(state)
        action = action.reshape(action.shape[0], 1)
        next_q = self.model_old(next_s).max(dim = 1)[0]
        reward_b = reward + self.GAMMA * next_q * torch.tensor(1 - ist).float()
        reward_b = reward_b.reshape(reward_b.shape[0], 1)
        L = self.loss(q.gather(1, action), reward_b)
        L.backward()
        self.opt.step()

    def update_q(self, state, action, reward, next_s, ist):
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
    
    def sampling(self, epoch):
        state = self.env.reset()
        init_state = state
        action = self.get_action(state)
        step = 0
        while True:
            step += 1
            next_s, reward, ist, _ = self.env.step(action)
            reward = self.reward_func(self.env, next_s, reward)
            self.update_q(state, action, torch.tensor(reward).float(), next_s, 1 if ist else 0)
            if ist:
                print('Epoch %6d, %6d steps' % (epoch, step), self.model_update(torch.tensor([init_state]).float())[0])
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
        
# MazeEnv
inputlen = 9
cnn = []
fc = [9, 100, 10, 4]
env = mazeenv.MazeEnv()

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
          epoch = 100000, replay = 2000, update_round = 100, reward_func = CartPole_reward_func)

dqn.main()