import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sac import SAC
from keypoints_encoder import KeypointsEncoder


class KeyQ(SAC):
    def __init__(self, cfg):
        self.encoder = KeypointsEncoder(cfg)
        with torch.no_grad():
            cfg['state_size'] = self.encoder(torch.zeros(1, *cfg['obs_shape']))[0].size(1)

        super().__init__(cfg)

        self.encoder.to(self.device)
        self.encoder_target = KeypointsEncoder(cfg).to(self.device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())

        critic_parameters = list(self.critic.parameters()) + self.encoder.critic_parameters()
        self.critic_optimizer = torch.optim.Adam(critic_parameters, lr=self.lr)

        actor_parameters = list(self.actor.parameters()) + self.encoder.actor_parameters()
        self.actor_optimizer = torch.optim.Adam(actor_parameters, lr=self.lr)

        if self.enable_decoder:
            ae_parameters = list(self.encoder.critic_parameters()) + list(self.encoder.decoder_parameters())
            self.decoder_optimizer = torch.optim.Adam(ae_parameters, lr=self.lr)

    def act(self, obs, sample=True):
        obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        obs /= 255.
        self.encoder.eval()
        with torch.no_grad():
            state = self.encoder.encode_actor(obs)
        return super().act(state, sample)

    def update_decoder(self, obs, action, next_obs):
        self.encoder.train()

        loss = self.encoder.compute_decoder_loss(obs, action, next_obs)

        self.decoder_optimizer.zero_grad()
        loss.backward()
        self.decoder_optimizer.step()

    def update_critic(self, obs, action, reward, next_obs, not_done):
        self.encoder.train()
        self.critic.train()
        alpha = self.log_alpha.exp().item()

        with torch.no_grad():
            next_actor_state, next_critic_state = self.encoder(next_obs)
            next_action, next_action_log_prob = self.actor.sample(next_actor_state)
            q1_next, q2_next = self.critic_target(next_critic_state, next_action)
            q_next = torch.min(q1_next, q2_next)
            value_next = q_next - alpha * next_action_log_prob
            q_target = reward + not_done * self.gamma * value_next

        critic_state = self.encoder.encode_critic(obs)
        q1, q2 = self.critic(critic_state, action)
        q1_loss = F.mse_loss(q1, q_target)
        q2_loss = F.mse_loss(q2, q_target)
        critic_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs):
        self.actor.train()

        actor_state, critic_state = self.encoder(obs)

        alpha = self.log_alpha.exp().item()
        action_new, action_new_log_prob = self.actor.sample(actor_state)

        q1_new, q2_new = self.critic(critic_state, action_new)
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
        super().update_target_networks()
        for target_param, param in zip(self.encoder_target.critic_parameters(), self.encoder.critic_parameters()):
            target_param.data.copy_((1.0-self.tau)*target_param.data + self.tau*param.data)

    def update(self):
        if self.enable_decoder and (len(self.replay_buffer) > self.batch_size) and (self.step % self.decoder_update_freq == 0):
            for _ in range(self.n_decoder_updates):
                obs, action, reward, next_obs, not_done = self.replay_buffer.sample(self.batch_size)
                obs = obs.float() / 255.
                next_obs = next_obs.float() / 255.
                self.update_decoder(obs, action, next_obs)

        if self.step < self.random_steps:
            return
        obs, action, reward, next_obs, not_done = self.replay_buffer.sample(self.batch_size)
        obs = obs.float() / 255.
        next_obs = next_obs.float() / 255.
        self.update_parameters(obs, action, reward, next_obs, not_done)

    def save(self):
        torch.save(self.encoder.state_dict(), self.checkpoint_dir+'/encoder.pth')
        torch.save(self.actor.state_dict(), self.checkpoint_dir+'/actor.pth')
        torch.save(self.critic.state_dict(), self.checkpoint_dir+'/critic.pth')

    def load(self):
        self.encoder.load_state_dict(torch.load(self.checkpoint_dir+'/encoder.pth'))
        self.actor.load_state_dict(torch.load(self.checkpoint_dir+'/actor.pth'))
        self.critic.load_state_dict(torch.load(self.checkpoint_dir+'/critic.pth'))

