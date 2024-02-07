import torch
import torch.nn as nn
from module.controller import DQNController
from module.vae import VAE
from module.mdnrnn import MDNRNN
import gymnasium as gym
from collections import deque
import random
import numpy as np
import utils
import os
from torch.utils.tensorboard import SummaryWriter


class ReplayBuffer(object):
    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, act, rew, next_obs, done, h, h_prime):
        self.buffer.append((obs, act, rew, next_obs, done, h, h_prime))

    def sample(self, batch_size: int = 32):
        batch = random.sample(self.buffer, batch_size)
        obs, act, rew, next_obs, done, h, h_prime = map(np.stack, zip(*batch))
        return obs, act, rew, next_obs, done, h, h_prime


def train(controller: DQNController, target_controller: DQNController, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, buffer: ReplayBuffer, batch_size: int = 32, gamma: float = 0.99, device: str = "cuda"):
    losses = []
    controller.train()
    for _ in range(10):
        obs, act, rew, next_obs, done, h, h_prime = buffer.sample(batch_size)
        latent_obs = torch.from_numpy(obs).float().to(device)
        act = torch.from_numpy(act).to(device)
        rew = torch.from_numpy(rew).float().to(device)
        latent_next_obs = torch.from_numpy(next_obs).float().to(device)
        done = torch.from_numpy(done).to(device)
        h = torch.from_numpy(h).float().to(device)
        h_prime = torch.from_numpy(h_prime).float().to(device)

        q = controller(latent_obs, h).squeeze()
        q = q.gather(1, act.unsqueeze(-1).to(torch.int64))
        with torch.no_grad():
            next_q = target_controller(
                latent_next_obs, h_prime).max(dim=-1)[0]
            target = rew.view(-1, 1) + done.view(-1, 1) * gamma * next_q
        loss = loss_fn(q.squeeze(), target.squeeze())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)


def evaluate(controller: DQNController, vae: VAE, rnn: MDNRNN, device: str = "cuda"):
    env = gym.make("CarRacing-v2", continuous=False)
    round = 10
    score = []
    stat = {}
    controller.eval()
    for _ in range(round):
        obs, _ = env.reset()
        total_reward = 0
        for _ in range(50):
            obs, _, _, _, _ = env.step(0)
        h, c = rnn.init_hidden()
        h, c = h.to(device), c.to(device)
        while True:
            obs_tensor = utils.image_array2tensor(utils.crop_frame(obs))
            latent_obs = vae.encode(obs_tensor.unsqueeze(0).to(device))
            a = controller.sample(latent_obs, h, 0)
            next_obs, rew, done, truncated, _ = env.step(a)
            _, h, c = rnn.sample(latent_obs, torch.tensor(
                [a]).view(-1, 1).to(device), h, c)
            total_reward += rew
            obs = next_obs
            if done or truncated:
                score.append(total_reward)
                break
    env.close()
    stat["score"] = np.array(score)
    stat["mean_score"] = np.mean(score)
    stat["std_score"] = np.std(score)
    return stat


def train_controller():
    env = gym.make("CarRacing-v2", continuous=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    buffer = ReplayBuffer(capacity=50000)
    frames_count = 0
    writer = SummaryWriter("log/controller2")
    max_frames = 200000000
    best_score = - np.inf

    # vae
    vae = VAE(img_channels=3, latent_size=32)
    vae.load_state_dict(torch.load("pretrained/vae2.pt"))
    vae.to(device)
    vae.eval()
    # mdnrnn
    rnn = MDNRNN(n_hidden=256, n_gaussians=5, latent_dim=32, action_dim=1)
    rnn.load_state_dict(torch.load("pretrained/mdnrnn.pt"))
    rnn.to(device)
    rnn.eval()
    # controller
    controller = DQNController(256 + 32, 128, env.action_space.n)
    controller.to(device)
    target_controller = DQNController(256 + 32, 128, env.action_space.n)
    target_controller.load_state_dict(controller.state_dict())
    target_controller.to(device)
    epsilon_scheduler = utils.Scheduler(0.5, 0.001, 2000000)

    optimizer = torch.optim.Adam(controller.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # training pipeline
    while True:
        obs, _ = env.reset()
        # skip the beginning of each rollout to skip the loading screen
        for _ in range(50):
            obs, _, _, _, _ = env.step(0)
        h, c = rnn.init_hidden()
        h, c = h.to(device), c.to(device)
        total_reward = 0
        while True:
            obs_tensor = utils.image_array2tensor(
                utils.crop_frame(obs)).unsqueeze(0).to(device)
            latent_obs = vae.encode(obs_tensor)
            a = controller.sample(latent_obs, h, epsilon_scheduler())
            next_obs, rew, done, truncated, _ = env.step(a)
            total_reward += rew
            next_obs_tensor = utils.image_array2tensor(
                utils.crop_frame(next_obs)).unsqueeze(0).to(device)
            latent_next_obs = vae.encode(next_obs_tensor)
            _, h_prime, c_prime = rnn.sample(
                latent_obs, torch.tensor([a]).view(-1, 1).to(device), h, c)
            buffer.push(latent_obs.cpu().numpy(), a, rew,
                        latent_next_obs.cpu().numpy(), 0 if done or truncated else 1, h.cpu().numpy(), h_prime.cpu().numpy())

            h, c = h_prime, c_prime
            obs = next_obs
            frames_count += 1

            if frames_count % 2000 == 0:
                loss = train(controller, target_controller,
                             optimizer, loss_fn, buffer, 512, 0.99, device)
                writer.add_scalar("train/loss", loss, frames_count)
            if frames_count % 4000 == 0:
                target_controller.load_state_dict(
                    controller.state_dict())
            if frames_count % 4000 == 0:
                stat = evaluate(target_controller, vae, rnn, device)
                mean_score = stat["mean_score"]
                writer.add_scalar(
                    "test/score", mean_score, frames_count)
                np.savez(
                    os.path.join("result", f"{frames_count:010d}"), score=stat["score"], mean_score=stat["mean_score"], std_score=stat["std_score"])
                if mean_score > best_score:
                    best_score = mean_score
                    torch.save(controller, "pretrained/controller2.pt")

            if done or truncated:
                print(f"train score: {total_reward}")
                break

        if frames_count > max_frames:
            break


if __name__ == "__main__":
    train_controller()
