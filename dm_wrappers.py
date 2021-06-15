from collections import deque, OrderedDict
import numpy as np

import dm_env
from dm_control.mujoco.engine import Camera


def xyz2pixels(xyz, cam_mat):
    """ Project 3D locations to pixel locations using the camera matrix """ 
    xyzs = np.ones((xyz.shape[0], xyz.shape[1]+1))
    xyzs[:, :xyz.shape[1]] = xyz
    xs, ys, s = cam_mat.dot(xyzs.T)
    x, y = xs/s, ys/s
    return x, y

def compute_pixel_offsets(cm_from, cm_to):
    """ Compute 2D pixel offsets caused by 2D camera movement """
    xyz0 = np.zeros((1, 3))
    x_from, y_from = xyz2pixels(xyz0, cm_from)
    x_to, y_to = xyz2pixels(xyz0, cm_to)
    return (x_to - x_from)[0], (y_to - y_from)[0]


class FlattenStateWrapper(dm_env.Environment):
    """ Flatten values in observations """
    def __init__(self, env):
        self._env = env
        self._observation_spec = OrderedDict()
        state_size = sum([np.int(np.prod(v.shape)) for v in env.observation_spec().values()])
        state_dtype = env.observation_spec
        self._observation_spec['state'] = dm_env.specs.Array(shape=(state_size,), dtype=np.float32, name='state')

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._env.action_spec()

    def _flatten_state(self, time_step):
        obs = [
            np.array([v]) if np.isscalar(v) else v.ravel()
            for v in time_step.observation.values()
        ]
        obs = np.concatenate(obs).astype(np.float32)
        return time_step._replace(observation=obs)

    def reset(self):
        time_step = self._env.reset()
        return self._flatten_state(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._flatten_state(time_step)

    def __getattr__(self, name):
        return getattr(self._env, name)


class KeypointsStateWrapper(dm_env.Environment):
    """ Represent state using keypoints in 2D cartesian coordinates """
    def __init__(self, env, relative_xy=True):
        self._env = env
        self.relative_xy = relative_xy

        xpos = env.physics.named.data.geom_xpos #[1:]
        state_size = xpos.shape[0] * 4

        self._observation_spec = OrderedDict()
        self._observation_spec['state'] = dm_env.specs.Array(shape=(state_size,), dtype=np.float32, name='state')

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._env.action_spec()

    def _get_state(self, time_step):
        self.camera_matrix = Camera(self.physics, height=84, width=84, camera_id=0).matrix

        def xyz2pixels_norm(xyz, cam_mat):
            x, y = xyz2pixels(xyz, cam_mat)
            x, y = x/42-1, y/42-1
            x[0] = 0
            return x, y

        xyz = self.physics.named.data.geom_xpos.copy()
        cam_mat = self.camera_matrix.copy()

        if len(self._xyzs) == 0:
            self._xyzs.extend([xyz, xyz, xyz])
            self._cam_mats.extend([cam_mat, cam_mat, cam_mat])
        else:
            self._xyzs.append(xyz)
            self._cam_mats.append(cam_mat)

        xyz_past = self._xyzs[-2]
        cam_mat_past = self._cam_mats[-2]

        x, y = xyz2pixels_norm(xyz, self.camera_matrix)
        x_past, y_past = xyz2pixels_norm(xyz_past, self.camera_matrix)

        x_vel = x - x_past
        y_vel = y - y_past

        if self.relative_xy:
            x = x - x.mean()
            y = y - y.mean()

        obs = np.concatenate([x, x_vel, y, y_vel])
        return time_step._replace(observation=obs.astype(np.float32))

    def reset(self):
        self._xyzs = []
        self._cam_mats = []
        time_step = self._env.reset()
        return self._get_state(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._get_state(time_step)

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionRepeatWrapper(dm_env.Environment):
    """ Repeat same action for k steps """
    def __init__(self, env, action_repeat=2):
        self._env = env
        self.action_repeat = action_repeat

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def step(self, action):
        reward = 0
        for _ in range(self.action_repeat):
            time_step = self._env.step(action)
            reward += time_step.reward
            if time_step.last():
                break
        return time_step._replace(reward=reward)

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    """
    Stack pixel observations from past k steps.
    Returns channel first.
    """
    def __init__(self, env, k=3):
        self._env = env
        self.k = k
        self._frames = deque([], maxlen=k)

        pixels = self._env.observation_spec()['pixels']
        obs_shape = (pixels.shape[2]*k, pixels.shape[0], pixels.shape[1])
        self._observation_spec = OrderedDict()
        self._observation_spec['pixels'] = dm_env.specs.Array(
            shape=obs_shape,
            dtype=pixels.dtype,
            name=pixels.name
        )
 
    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        time_step = self._env.reset()
        for _ in range(self.k):
            self._frames.append(time_step.observation['pixels'])
        return self._get_obs(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        self._frames.append(time_step.observation['pixels'])
        return self._get_obs(time_step)

    def _get_obs(self, time_step):
        obs = np.concatenate(self._frames, 2).transpose(2, 0, 1)
        return time_step._replace(observation=obs)

    def __getattr__(self, name):
        return getattr(self._env, name)


class CameraOffsetFrameStackWrapper(dm_env.Environment):
    """
    Stack pixel observations from past k steps.
    Also, encode camera offset information in channel 0.
    Returns channel first.
    """
    def __init__(self, env, k=3):
        self._env = env
        self.k = k
        self._frames = deque([], maxlen=k)
        self._camera_matrices = deque([], maxlen=k)

        pixels = self._env.observation_spec()['pixels']
        obs_shape = (pixels.shape[2]*k+1, pixels.shape[0], pixels.shape[1])
        self._observation_spec = OrderedDict()
        self._observation_spec['pixels'] = dm_env.specs.Array(
            shape=obs_shape,
            dtype=pixels.dtype,
            name=pixels.name
        )

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        time_step = self._env.reset()
        camera_matrix = Camera(self.physics, height=84, width=84, camera_id=0).matrix.copy()
        for _ in range(self.k):
            self._frames.append(time_step.observation['pixels'])
            self._camera_matrices.append(camera_matrix)
        return self._get_obs(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        self._frames.append(time_step.observation['pixels'])
        camera_matrix = Camera(self.physics, height=84, width=84, camera_id=0).matrix.copy()
        self._camera_matrices.append(camera_matrix)
        return self._get_obs(time_step)

    def _get_obs(self, time_step):
        def scale_diff(diff, scale):
            diff = diff / scale # to -1..1
            diff = int(np.floor(diff * 255)) # to -255..255
            return diff

        def unscale_diff(diff, scale):
            diff = diff / 255
            diff = diff * scale
            return diff

        camera_diffs = np.zeros((1, 84, 84), dtype=np.uint8)
        for i, cam_mat in enumerate(self._camera_matrices):
            x_diff, y_diff = compute_pixel_offsets(cam_mat, self._camera_matrices[-1])
            assert x_diff < 84
            assert y_diff < 84
            x_diff = scale_diff(x_diff, 84)
            y_diff = scale_diff(y_diff, 84)
            if x_diff > 0:
                camera_diffs[:, i, 0] = x_diff
            else:
                camera_diffs[:, i, 1] = -x_diff
            if y_diff > 0:
                camera_diffs[:, i, 2] = y_diff
            else:
                camera_diffs[:, i, 3] = -y_diff

        obs = np.concatenate(self._frames, 2).transpose(2, 0, 1)
        obs = np.concatenate([camera_diffs, obs], 0)
        return time_step._replace(observation=obs)

    def __getattr__(self, name):
        return getattr(self._env, name)

