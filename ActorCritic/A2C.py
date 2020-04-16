import torch
import numpy as np
import pdb

from common.utils import cuda, Fake_TXSW

class A2C:
    def __init__(self,
                 model,
                 threads,
                 learning_rate,
                 a2c_entropy_beta,
                 a2c_value_beta,
                 a2c_clip_grad,
                 **kwargs
                ):
        self.model = model
        self.threads = threads
        self.LR = learning_rate
        self.opt = torch.optim.Adam(self.model.parameters(), learning_rate)
        self.ENTROPY_BETA = a2c_entropy_beta
        self.VALUE_BETA = a2c_value_beta
        self.CLIP_GRAD = a2c_clip_grad

    def update_policy(self, rollouts, now_frame = None, TXSW = None):
        self.model.train()
        state, action, reward, next_state, returns = rollouts.get_flatten('obs', 'actions', 'rewards', rollouts.obs[1:], 'returns')
        self.opt.zero_grad()
        state = cuda(state)
        action = cuda(action)
        reward = cuda(returns)
        next_state = cuda(next_state)
        dist, v = self.model(state)
        log_s_dist = torch.nn.functional.log_softmax(dist, dim = 1)
        s_dist = torch.nn.functional.softmax(dist, dim = 1)
        pd_loss = -((reward - v.detach()) * torch.gather(log_s_dist, 1, action)).mean()
        entropy_loss = (s_dist * log_s_dist).sum(dim=1).mean()
        v_loss = torch.nn.functional.mse_loss(reward, v)
        if now_frame != None:
            TXSW.add_scalar('pd_loss', pd_loss.item(), now_frame)
            TXSW.add_scalar('entropy_loss', entropy_loss.item(), now_frame)
            TXSW.add_scalar('v_loss', v_loss.item(), now_frame)
            TXSW.add_scalar('v_mean', v.mean().item(), now_frame)
        loss = pd_loss + self.ENTROPY_BETA * entropy_loss + self.VALUE_BETA * v_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.CLIP_GRAD)
        self.opt.step()
        return loss.item()
