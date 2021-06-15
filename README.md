# KeyQ: End-to-End Learning of Keypoint Representations for Continuous Control from Images

This is a PyTorch implementation of the **KeyQ** method proposed in the paper "End-to-End Learning of Keypoint Representations for Continuous Control from Images".

## Dependencies

The main dependencies are `pytorch`, `numpy` and `dm_control`.

Install the required dependencies by creating an anaconda environment from `conda_env.yml`:
```
conda env create -f conda_env.yml
```
and then activate the installed `keyq` environment: `conda activate keyq`

Training KeyQ on the [DeepMind Control Suite](https://github.com/deepmind/dm_control) requires a valid [MuJoCo](http://www.mujoco.org/) installation. Refer to [https://github.com/deepmind/dm_control#requirements-and-installation](https://github.com/deepmind/dm_control#requirements-and-installation) for instructions on installing MuJoCo.

## Experiments on PlaNet Benchmark

KeyQ results reported in the paper (for the six tasks in the PlaNet benchmark) can be reproduced by running:
```
python train.py domain=ball_in_cup task=catch relative_xy=False
python train.py domain=cartpole task=swingup
python train.py domain=cheetah task=run train_episodes=1000
python train.py domain=finger task=spin
python train.py domain=reacher task=easy lr=1e-3
python train.py domain=walker task=walk train_episodes=1000
```
