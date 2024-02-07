import gymnasium as gym
from gymnasium.wrappers import TransformObservation
import numpy as np
from gymnasium import ObservationWrapper
import utils
from module.vae import VAE
from module.mdnrnn import MDNRNN
from PIL import Image
import torch


vae = VAE(3, 32)
vae.load_state_dict(torch.load("pretrained/vae2.pt"))
rnn = MDNRNN(256, 5, 32, 1)
rnn.load_state_dict(torch.load("pretrained/mdnrnn.pt"))


def transform(obs):
    obs = utils.crop_frame(obs)
    obs = utils.image_array2tensor(obs).unsqueeze(0)
    obs = vae.encode(obs).numpy().squeeze()
    return obs


class TransformObservation(ObservationWrapper):
    def __init__(self, env, transform, rnn=rnn):
        super(TransformObservation, self).__init__(env)
        self.transform = transform
        self.observation_space = gym.spaces.Box(
            0, 255, (64, 64, 3), np.uint8
        )
        self.rnn = rnn
        h, c = self.rnn.init_hidden()

    def observation(self, obs):
        latent_obs = self.transform(obs)


env = gym.make("CarRacing-v2", continuous=False)
env = TransformObservation(env, transform)
obs, _ = env.reset()
print(obs)
print(obs.shape)
