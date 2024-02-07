from module.vae import VAE
from module.mdnrnn import MDNRNN
from module.controller import A2CController
import os
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import utils


def train(a2c, obs_traj, act_traj, rew_traj, h_traj, actor_optim, critic_optim, loss_fn, device):
    G = 0.0
    G_traj = []
    a2c.train()
    for rew in rew_traj[::-1]:
        G = rew + 0.99 * G
        G_traj.insert(0, G)

    obs_traj = torch.from_numpy(
        np.stack(obs_traj)).float().to(device).squeeze()
    h_traj = torch.from_numpy(np.stack(h_traj)).float().to(device).squeeze()
    act_traj = torch.from_numpy(
        np.stack(act_traj)).float().to(device).view(-1, 1)
    G_traj = torch.from_numpy(np.stack(G_traj)).float().to(device).view(-1, 1)

    prob, v = a2c(obs_traj, h_traj, act_traj)
    actor_optim.zero_grad()

    actor_loss = - \
        torch.log(prob.gather(-1, act_traj.to(torch.int64))).squeeze() * \
        (G_traj - v.detach()).squeeze()
    assert G_traj.shape == v.shape, f"{G_traj.shape} != {v.shape}"
    actor_loss = actor_loss.sum()
    actor_loss.backward()
    actor_optim.step()

    # update critic
    critic_optim.zero_grad()
    critic_loss = loss_fn(v.squeeze(), G_traj.squeeze())
    critic_loss.backward()
    critic_optim.step()

    return actor_loss.item(), critic_loss.item()


def evaluate(controller: A2CController, vae: VAE, rnn: MDNRNN):
    env = gym.make("CarRacing-v2", continuous=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    round = 20
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
    frames_count = 0
    writer = SummaryWriter("log/a2c")
    episodes = 10000
    best_score = - np.inf

    # vae
    vae = VAE(img_channels=3, latent_size=32).to(device)
    vae.load_state_dict(torch.load("pretrained/vae2.pt"))
    vae.to(device)
    vae.eval()

    # mdnrnn
    rnn = MDNRNN(n_hidden=256, n_gaussians=5,
                 latent_dim=32, action_dim=1).to(device)
    rnn.load_state_dict(torch.load("pretrained/mdnrnn.pt"))
    rnn.to(device)
    rnn.eval()

    # controller
    controller = A2CController(256 + 32, 256, env.action_space.n).to(device)
    actor_optimizer = torch.optim.Adam(controller.actor.parameters(), lr=1e-3)
    critic_optimizer = torch.optim.Adam(
        controller.critic.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    epsilon_scheduler = utils.Scheduler(0.5, 0.001, 2000000)

    # training pipeline
    for i in range(episodes):
        obs, _ = env.reset()
        for _ in range(50):
            obs, _, _, _, _ = env.step(0)

        h, c = rnn.init_hidden()
        h, c = h.to(device), c.to(device)
        total_reward = 0
        obs_traj = []
        act_traj = []
        rew_traj = []
        next_obs_traj = []
        done_traj = []
        h_traj = []
        h_prime_traj = []
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

            obs_traj.append(latent_obs.cpu().numpy())
            act_traj.append(a)
            rew_traj.append(rew)
            next_obs_traj.append(latent_next_obs.cpu().numpy())
            done_traj.append(0 if done or truncated else 1)
            h_traj.append(h.cpu().numpy())
            h_prime_traj.append(h_prime.cpu().numpy())

            h, c = h_prime, c_prime
            obs = next_obs
            frames_count += 1

            if done or truncated:
                print(f"{i} score: {total_reward}")
                writer.add_scalar("train/score", total_reward, i)
                break
        # training
        actor_loss, critic_loss = train(controller, obs_traj, act_traj, rew_traj, h_traj,
                                        actor_optimizer, critic_optimizer, loss_fn, device)
        writer.add_scalar("train/actor_loss", actor_loss, i)
        writer.add_scalar("train/critic_loss",
                          critic_loss, i)
        # evaluation
        stat = evaluate(controller, vae, rnn)
        writer.add_scalar(
            "test/score", stat["mean_score"], i)
        if best_score < stat["mean_score"]:
            best_score = stat["mean_score"]
            torch.save(controller.state_dict(), "pretrained/a2c.pt")
        np.savez(os.path.join("a2c_result", f"{i:08d}"),
                 score=stat["score"], mean_score=stat["mean_score"], std_score=stat["std_score"])


if __name__ == "__main__":
    train_controller()
