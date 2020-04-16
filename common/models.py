import numpy as np
import torch
import torch.nn as nn
from common.distributions import FixedCategorical

class AtariCNN(nn.Module):
    def __init__(self, inputlen, recurrent=False, hidden_size=512):
        super(AtariCNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(inputlen, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU()
        )
        self.fc = nn.Sequential(nn.Linear(64 * 7 * 7, hidden_size), nn.ReLU())

    def forward(self, x):
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

#input: state; output: policy distribution, value function
class ActorCriticNet(torch.nn.Module):
    def __init__(self, inputlen, featurelen, action_space, feature_net):
        super(ActorCriticNet, self).__init__()
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

    def get_action(self, state):
        #pdb.set_trace()
        #state = cuda(torch.tensor(np.expand_dims(state, 0)).float())
        with torch.no_grad():
            policy = self.forward(state, apply_softmax = True)[0]
        dist = FixedCategorical(policy)
        return dist.sample()