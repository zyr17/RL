import torch
import numpy as np
import random
import pickle

class MazeEnv:

    def __init__(self):

        self.MAP = [
            "SXE",
            ".X.",
            "..."
        ]
        self.ACT = [ 0, 1, 2, 3 ] # left right up down
        self.ACTDELTA = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.INIT_STATE = 0
        def find_start():
            for i in self.MAP:
                for j in i:
                    if j == 'S':
                        return
                    self.INIT_STATE += 1
        find_start()
        self.step_results = []

    def get_init_state(self):
        return self.statenum2state(self.INIT_STATE)

    def statenum2state(self, num):
        return np.array(np.array(range(len(self.MAP) * len(self.MAP[0]))) == num, dtype='float')

    def get_reward(self, state, act):
        state = np.argmax(state)
        x = state // len(self.MAP[0])
        y = state % len(self.MAP[0])
        xx = x + self.ACTDELTA[act][0]
        yy = y + self.ACTDELTA[act][1]
        if xx < 0 or xx >= len(self.MAP) or yy < 0 or yy >= len(self.MAP[0]): # hit border
            return self.statenum2state(x * len(self.MAP[0]) + y), -1
        if self.MAP[xx][yy] == 'X': # hit wall
            return self.statenum2state(x * len(self.MAP[0]) + y), -1
        if self.MAP[xx][yy] == 'E': # enter exit
            return self.statenum2state(xx * len(self.MAP[0]) + yy), 1
        # normal move
        return self.statenum2state(xx * len(self.MAP[0]) + yy), -1

    def is_terminal(self, state, epoch, step, print_step = False, print_terminal = True, time_delay = 0):
        state = np.argmax(state)
        x = state // len(self.MAP[0])
        y = state % len(self.MAP[0])
        ist = self.MAP[x][y] == 'E'
        if ist:
            if print_terminal:
                print('Epoch %3d, %4d steps' % (epoch, step))
            self.step_results.append(step)
        elif print_step:
            p = [x for x in self.MAP]
            p[x] = p[x][:y] + '*' + p[x][y + 1:]
            print('Step %d:\n%s' % (step, '\n'.join(p)))
            time.sleep(time_delay)
        return ist

    def random_action(self, state):
        return random.randint(0, len(self.ACT) - 1)

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
            lastfea = i[0]
        self.fc = torch.nn.ModuleList()
        for i, j in zip(fc[:-1], fc[1:]):
            self.fc.append(torch.nn.Linear(i, j))
            #torch.nn.Dropout(0.5),
            self.fc.append(torch.nn.ReLU())
        self.fc = self.fc[:-1]

    def forward(self, inputs):
        x = inputs
        for cnn in self.cnn:
            x = cnn(x)
            #print(x.shape)
        x = x.reshape(-1)
        for fc in self.fc:
            x = fc(x)
            #print(x.shape)
        return x

class DQN:
    def __init__(self, env, inputlen, cnn, fc, 
                 alpha = 0.1, gamma = 0.95, eps = 0.9, 
                 epoch = 1000, replay = 1000000, update_round = 10000,
                 learning_rate = 0.001
                ):
        self.env = env
        self.ALPHA = alpha
        self.GAMMA = gamma
        self.EPS = eps
        self.EPOCH = epoch
        self.REPLAY = replay
        self.UPDATE = update_round
        self.model_run = DQNnet(inputlen, cnn, fc)
        self.model_update = DQNnet(inputlen, cnn, fc)
        self.model_run.load_state_dict(self.model_update.state_dict())
        self.update_count = 0
        self.replay_count = 0
        self.replay = []
        self.LR = learning_rate
        self.opt = torch.optim.Adam(self.model_update.parameters(), learning_rate)
        self.loss = torch.nn.MSELoss()
        self.model_run.eval()
        self.model_update.train()

    def get_action(self, state, eps, action = None, q = None):
        if random.random() > eps:
            if action == None:
                q = self.model_run(torch.tensor(state).float())
            return self.env.random_action(state), q
        if action != None:
            return action, q
        q = self.model_run(torch.tensor(state).float())
        return torch.argmax(q).item(), q

    def real_update_q(self, state, action, reward):
        #print('real update q', state, action, reward)
        self.opt.zero_grad()
        q = self.model_update(torch.tensor(state).float())
        L = self.loss(q[action], reward)
        L.backward()
        self.opt.step()

    def update_q(self, state, action, reward):
        if len(self.replay) < self.REPLAY:
            self.replay.append([state, action, reward])
        else:
            self.replay_count = (self.replay_count + 1) % self.REPLAY
            self.replay[self.replay_count] = [state, action, reward]
            self.real_update_q(*self.replay[random.randint(0, self.REPLAY - 1)])
            #self.real_update_q(*self.replay[self.replay_count])
            self.update_count = (self.update_count + 1) % self.UPDATE
            if self.update_count == 0:
                #print('update model')
                self.model_run.load_state_dict(self.model_update.state_dict())
    
    def sampling(self, epoch):
        state = self.env.get_init_state()
        action, state_q = self.get_action(state, self.EPS)
        step = 0
        while True:
            step += 1
            next_s, reward = self.env.get_reward(state, action)
            next_a, next_s_q = self.get_action(next_s, 1)
            ist = self.env.is_terminal(next_s, epoch, step)
            if ist:
                self.update_q(state, action, torch.tensor(reward).float())
            else:
                delta = reward + self.GAMMA * next_s_q[next_a] - state_q[action]
                state_q[action] += self.ALPHA * delta
                self.update_q(state, action, torch.tensor(state_q[action].item()))
            if ist:
                for i in range(9):
                    i = torch.tensor(np.array(np.array(range(9)) == i, dtype='float')).float()
                    print(self.model_update(i))
                break
            state = next_s
            action, state_q = self.get_action(state, self.EPS, next_a, next_s_q)
    
    def main(self):
        for ep in range(self.EPOCH):
            self.sampling(ep)
        
inputlen = 9
cnn = []
fc = [9, 100, 10, 4]

dqn = DQN(MazeEnv(), inputlen, cnn, fc, replay = 1000, update_round = 1)

dqn.main()