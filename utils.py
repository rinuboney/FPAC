import numpy as np
import torch.nn as nn
from dm_control.mujoco.engine import Camera


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

def gt_keypoints(env):
    """ Extract 2D pixel locations of objects in the environment """
    camera_matrix = Camera(env.physics, height=84, width=84, camera_id=0).matrix

    def xyz2pixels(xyz):
        xyzs = np.ones((xyz.shape[0], xyz.shape[1]+1))
        xyzs[:, :xyz.shape[1]] = xyz
        xs, ys, s = camera_matrix.dot(xyzs.T)
        return xs/s, ys/s

    xyz = env.physics.named.data.geom_xpos[1:]
    return xyz2pixels(xyz)


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, hidden_layers=1, activation_fn=nn.SiLU, init_w=3e-3):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size), activation_fn()]
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(hidden_size, hidden_size), activation_fn()])
        layers.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*layers)
        if init_w is not None:
            self.net[-1].weight.data.uniform_(-init_w, init_w)
            self.net[-1].bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        return self.net(x)

