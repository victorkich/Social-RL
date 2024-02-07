# README

This repo is a reimplementation of _World Model_(https://arxiv.org/pdf/1803.10122.pdf).

Currently it's tailored for CarRacing task as described in _Procedure 3.2_. We train all parts separately. Different from original paper that they use `CMA-ES`, we leverages RL algorithms as controller.

## Usage Guide

### Collect Rollouts

Initially, we utilize a pretrained DQN-based CarRacing policy to gather data for later stages. We collect 1000 rollouts due to constraints in resources and time.

```bash
cd rl-car-racing && python main.py play 1000 false
```

The collected rollouts are in `dataset/` directory with numpy npz file format. Each file encapsulates (_states_, _rews_, _act_, _next_states_, _done_mask_). Additionally, the root directory contains a `meta.json` file which provides metadata for each rollout, including (_frames_per_episode_, _score_, _record_filename_).

### Train VAE

For training the Variational Autoencoder (VAE), we utilize a subset of 200 rollouts. This pretrains the VAE model effectively.

```bash
python dataset_generate.py --npz2img --start 0 --end 200 --src dataset --dst vae_dataset
```

### Train MDN-RNN

The MDN-RNN training begins with encoding raw images into latent representations using the pretrained VAE model. This process accelerates the MDN-RNN training by utilizing these pre-encoded latent vectors.

```bash
python dataset_generate.py --raw2latent --start 0 --end 1000 --src dataset --dst vae_dataset
```

### Train Controller

```bash
python train_controller.py
```
