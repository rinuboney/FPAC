import sys
import random
import numpy as np
import torch

from dm_control import suite
from dm_control.mujoco.engine import Camera
from dm_control.suite.wrappers import pixels
from dm_wrappers import *

def rollout(agent, env, train=False, random=False):
    spec = env.action_spec()
    time_step = env.reset()
    obs = time_step.observation
    episode_return = 0
    while not time_step.last():
        if random or (agent.step < agent.random_steps):
            action = np.random.uniform(spec.minimum, spec.maximum, spec.shape)
        else:
            action = agent.act(obs, sample=train)

        time_step = env.step(action)
        next_obs = time_step.observation
        episode_return += time_step.reward

        if train:
            agent.replay_buffer.append(obs, action, time_step.reward, next_obs, time_step.discount)
            agent.update()
            agent.step += 1

        obs = next_obs

    return episode_return

def evaluate(agent, env, n_episodes=10):
    returns = [
        rollout(agent, env, train=False, random=False)
        for i in range(n_episodes)
    ]
    return np.mean(returns)

def train(agent, env, n_episodes=1000):
    for episode in range(n_episodes):
        train_return = rollout(agent, env, train=True)
        print(f'Episode {episode}. Return {train_return}')

        if (episode+1) % 20 == 0:
            eval_return = evaluate(agent, env)
            print(f'Eval Reward {eval_return}')

        if agent.domain == 'reacher' and episode == 199:
            print('Changing learning rate to 3e-4')
            agent.critic_optimizer.param_groups[0]['lr'] = 3e-4
            agent.actor_optimizer.param_groups[0]['lr'] = 3e-4
            agent.alpha_optimizer.param_groups[0]['lr'] = 3e-4

def load_env(cfg):
    env = suite.load(cfg['domain'], cfg['task'], task_kwargs={'random': cfg['seed']})
    env = ActionRepeatWrapper(env, cfg['action_repeat'])

    if cfg['observation'] == 'pixels':
        env = pixels.Wrapper(env, render_kwargs={'height': 84, 'width': 84, 'camera_id': 0})
        if cfg['use_camera_offset']:
            env = CameraOffsetFrameStackWrapper(env, cfg['frame_stack'])
        else:
            env = FrameStackWrapper(env, cfg['frame_stack']) 
        cfg['obs_shape'] = env.observation_spec()['pixels'].shape
    elif cfg['observation'] == 'default':
        env = FlattenStateWrapper(env)
        cfg['obs_shape'] = env.observation_spec()['state'].shape
    elif cfg['observation'] == 'keypoints':
        env = KeypointsStateWrapper(env, cfg['relative_xy'])
        cfg['obs_shape'] = env.observation_spec()['state'].shape
    else:
        raise Exception('Unknown observation space')

    cfg['act_size'] = env.action_spec().shape[0]
    return env, cfg

if __name__ == '__main__':
    cfg = dict(
        seed = random.randint(1, 99),

        # Task
        domain = 'cartpole',
        task = 'swingup',
        observation = 'pixels',
        group = 'corl_cleanup',
        frame_stack = 2,
 
        # SAC
        train_episodes = 500,
        random_steps = 1000,
        batch_size = 128,
        lr = 3e-4,
        replay_buffer_capacity = 10**5,
        gamma = 0.99,
        tau = 0.01,
        init_temperature = 0.1,
        actor_update_freq = 2,
        target_update_freq = 2,

        hidden_layers = 2,
        hidden_size = 1024,

        # FPAC
        num_keypoints = 32,
        keypoint_temperature = 0.5,
        use_camera_offset = True,
        relative_xy = True,
        enable_decoder = False,
    )

    cfg_args = {}
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            key, val = arg.split('=')
            assert key in cfg, f'{key} not in cfg'
            if key not in ['domain', 'task', 'group', 'observation']:
                val = eval(val)
            cfg_args[key] = val
    cfg.update(cfg_args)

    # Set action_repeat used in PlaNet benchmark
    cfg['action_repeat'] = {
        'cartpole': 8,
        'reacher': 4,
        'cheetah': 4,
        'finger': 2,
        'ball_in_cup': 4,
        'walker': 2
    }[cfg['domain']]

    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])

    env, cfg = load_env(cfg)

    if cfg['observation'] == 'pixels':
        from fpac import FPAC
        agent = FPAC(cfg)
    else:
        from sac import SAC
        agent = SAC(cfg)

    train(agent, env, n_episodes=agent.train_episodes)

