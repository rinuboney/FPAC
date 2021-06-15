import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from utils import weight_init

def compute_keypoints_1d(key_logits, coords=None, temperature=1):
    if coords is None:
        coords_size = key_logits.size(-1)
        coords = torch.linspace(-1, 1, coords_size, device=key_logits.device)[None, None, :]

    def compute_coord(other_axis):
        logits_1d = key_logits.mean(other_axis)
        key_probs = F.softmax(logits_1d / temperature, dim=-1)
        mean = (coords * key_probs).sum(-1)
        return mean, key_probs

    h_mean, h_probs = compute_coord(-1)
    w_mean, w_probs = compute_coord(-2)

    presence = torch.tanh(torch.flatten(key_logits, -2, -1).mean(-1))

    return h_mean, w_mean, presence, h_probs, w_probs

def compute_heatmaps(h_mean, w_mean, coords, presence=None, std=0.1):
    h_mean, w_mean = h_mean[:, :, None, None], w_mean[:, :, None, None]
    h, w = coords[:, :, :, None], coords[:, :, None, :]
    g_h = (h - h_mean)**2
    g_w = (w - w_mean)**2
    dist = (g_h + g_w) / (std**2)
    g_hw = torch.exp(-dist)
    if presence is not None:
        g_hw = g_hw * presence[:, :, None, None]
    return g_hw


class KeypointsEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        for k, v in cfg.items():
            setattr(self, k, v)

        self.num_filters = 32
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, self.num_filters, 3, stride=2), nn.SiLU(),
            nn.Conv2d(self.num_filters, self.num_filters, 3), nn.SiLU(),
            nn.Conv2d(self.num_filters, self.num_filters, 3), nn.SiLU(),
            nn.Conv2d(self.num_filters, self.num_filters, 3), nn.SiLU(),
            nn.Conv2d(self.num_filters, self.num_keypoints, 1)
        )

        with torch.no_grad():
            coords_size = self.encoder_conv(torch.zeros(1, 3, 84, 84)).size(-1)
        self.register_buffer('coords', torch.linspace(-1, 1, coords_size)[None, None, :])

        if self.enable_decoder:
            self.decoder_conv = nn.Sequential(
                nn.ConvTranspose2d(self.num_keypoints*2, self.num_filters, 3), act_fn(),
                nn.ConvTranspose2d(self.num_filters, self.num_filters, 3), act_fn(),
                nn.ConvTranspose2d(self.num_filters, self.num_filters, 3), act_fn(),
                nn.ConvTranspose2d(self.num_filters, 3, 3, stride=2, output_padding=1)
            )

            self.style_conv = nn.Sequential(
                nn.Conv2d(3, self.num_filters, 3, stride=2), act_fn(),
                nn.Conv2d(self.num_filters, self.num_filters, 3), act_fn(),
                nn.Conv2d(self.num_filters, self.num_filters, 3), act_fn(),
                nn.Conv2d(self.num_filters, self.num_keypoints, 3)
            )

        self.apply(weight_init)

    def actor_parameters(self):
        return []

    def critic_parameters(self):
        return list(self.encoder_conv.parameters())

    def decoder_parameters(self):
        return list(self.decoder_conv.parameters()) + list(self.style_conv.parameters())

    def forward(self, x):
        if self.use_camera_offset:
            # Extract camera offset information encoded in channel 0
            x_diff = (x[:, 0, :self.frame_stack, 0] - x[:, 0, :self.frame_stack, 1]) * 84 # offset in 84x84 pixel space, range [-84, 84]
            y_diff = (x[:, 0, :self.frame_stack, 2] - x[:, 0, :self.frame_stack, 3]) * 84
            x_diff = x_diff / 42. # diff in -1..1 feature space, range [-2, 2]
            y_diff = y_diff / 42.
            x = x[:, 1:, :, :]

        batch_size = x.size(0)
        x = x.reshape(batch_size*self.frame_stack, 3, x.size(-2), x.size(-1))
        key_logits = self.encoder_conv(x)

        h_mean, w_mean, presence, h_probs, w_probs = compute_keypoints_1d(key_logits, self.coords, self.keypoint_temperature)

        self.key_logits = key_logits.view(batch_size, self.frame_stack, self.num_keypoints, key_logits.size(-2), key_logits.size(-1))
        self.h_mean = h_mean = h_mean.view(batch_size, self.frame_stack, self.num_keypoints)
        self.w_mean = w_mean = w_mean.view(batch_size, self.frame_stack, self.num_keypoints)
        self.presence = presence = presence.view(batch_size, self.frame_stack, -1)

        self.h_probs = h_probs.view(batch_size, self.frame_stack, self.num_keypoints, key_logits.size(-1))
        self.w_probs = w_probs.view(batch_size, self.frame_stack, self.num_keypoints, key_logits.size(-1))

        if self.use_camera_offset:
            # shift past keypoints so that all are in the current camera frame
            h_mean = h_mean + y_diff[:, :, None]
            w_mean = w_mean + x_diff[:, :, None]

        z = torch.cat([presence, h_mean, w_mean], -1)
        z_diff = z[:, 1:] - z[:, :-1]

        if self.relative_xy:
            w_mean = w_mean - w_mean.mean(-1, keepdim=True)
            h_mean = h_mean - h_mean.mean(-1, keepdim=True)

        z = torch.cat([presence, h_mean, w_mean], -1)
        z = torch.cat([z[:, -1:], z_diff], 1)
        z = torch.flatten(z, 1, -1)
        return z.detach(), z

    def encode_actor(self, obs):
        with torch.no_grad():
            z, _ = self.forward(obs)
        return z

    def encode_critic(self, obs):
        _, z = self(obs)
        return z

    def compute_decoder_loss(self, obs, act, next_obs):
        obs = obs[:, -3:]
        next_obs = next_obs[:, -3:]
        obs_style = next_obs[torch.randperm(obs.size(0))]

        key_logits = self.encoder_conv(obs)
        h_mean, w_mean, presence, _, _ = compute_keypoints_1d(key_logits, self.coords, self.keypoint_temperature)
        if not self.keypoint_presence:
            presence = None
        self.heatmaps = heatmaps = compute_heatmaps(h_mean, w_mean, self.coords, presence)

        style_features = self.style_conv(obs_style)
        features = torch.cat([heatmaps, style_features], 1)

        pred = self.decoder_conv(features)
        self.reconstruction = torch.sigmoid(pred)

        loss = F.binary_cross_entropy_with_logits(pred, obs)
        return loss

