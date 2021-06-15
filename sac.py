import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from replay_buffer import ReplayBuffer
from utils import MLP


class Critic(nn.Module):
    """ Twin Q-networks """
    def __init__(self, cfg):
        super().__init__()
        self.net1 = MLP(cfg['state_size']+cfg['act_size'], 1, cfg['hidden_size'], cfg['hidden_layers'])
        self.net2 = MLP(cfg['state_size']+cfg['act_size'], 1, cfg['hidden_size'], cfg['hidden_layers'])

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        return self.net1(state_action), self.net2(state_action)


class Actor(nn.Module):
    """ Gaussian Policy """
    def __init__(self, cfg):
        super().__init__()
        self.act_size = cfg['act_size']
        self.net = MLP(cfg['state_size'], cfg['act_size']*2, cfg['hidden_size'], cfg['hidden_layers'])

    def forward(self, state):
        x = self.net(state)
        mean, log_std = x[:, :self.act_size], x[:, self.act_size:]
        log_std = torch.clamp(log_std, min=-10, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        normal = Normal(mean, log_std.exp())
        x = normal.rsample()

        # Enforcing action bounds
        action = torch.tanh(x)
        log_prob = normal.log_prob(x) - torch.log(1 - action**2 + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob


class SAC:
    def __init__(self, cfg):
        self.cfg = cfg
        for k, v in cfg.items():
            setattr(self, k, v)
        if 'state_size' not in cfg:
            cfg['state_size'] = self.obs_shape[0]

        if 'device' in cfg:
            self.device = torch.device(self.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.critic = Critic(cfg).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.critic_target = Critic(cfg).to(self.device)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.actor = Actor(cfg).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        self.target_entropy = -self.act_size
        self.log_alpha = torch.tensor(np.log(self.init_temperature), requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)

        self.step = 0
        self.replay_buffer = ReplayBuffer(capacity=self.replay_buffer_capacity, device=self.device)

    def act(self, obs, sample=True):
        self.actor.eval()
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        with torch.no_grad():
            mean, log_std = self.actor(obs)
        if sample:
            normal = Normal(mean, log_std.exp())
            x = normal.rsample()
        else:
            x = mean
        action = torch.tanh(x)
        return action[0].detach().cpu().numpy()

    def update_critic(self, state, action, reward, next_state, not_done):
        self.critic.train()
        alpha = self.log_alpha.exp().item()

        with torch.no_grad():
            next_action, next_action_log_prob = self.actor.sample(next_state)
            q1_next, q2_next = self.critic_target(next_state, next_action)
            q_next = torch.min(q1_next, q2_next)
            value_next = q_next - alpha * next_action_log_prob
            q_target = reward + not_done * self.gamma * value_next

        q1, q2 = self.critic(state, action)
        q1_loss = 0.5*F.mse_loss(q1, q_target)
        q2_loss = 0.5*F.mse_loss(q2, q_target)
        critic_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, state):
        self.actor.train()
        alpha = self.log_alpha.exp().item()
        action_new, action_new_log_prob = self.actor.sample(state)
        q1_new, q2_new = self.critic(state, action_new)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (alpha*action_new_log_prob - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (action_new_log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

    def update_target_networks(self):
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_((1.0-self.tau)*target_param.data + self.tau*param.data)

    def update_parameters(self, state, action, reward, next_state, not_done):
        self.update_critic(state, action, reward, next_state, not_done)

        if self.step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(state)

        if self.step % self.target_update_freq == 0:
            self.update_target_networks()

    def update(self):
        if self.step < self.random_steps:
            return
        batch = self.replay_buffer.sample(self.batch_size)
        self.update_parameters(*batch)

