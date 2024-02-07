import torch
import torch.nn as nn
import random


class A2C(nn.Module):
    def __init__(self, obs_dim: int = 96 * 96 * 3, hidden_dim: int = 256, act_dim: int = 5):
        super(A2C, self).__init__()
        self.hidden_dim = hidden_dim
        self.act_dim = act_dim
        self.obs_dim = obs_dim

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, act):
        return self.actor(obs), self.critic(torch.cat([obs, act], dim=-1))

    def sample_action(self, obs, epsilon=0.001):
        if random.random() < epsilon:
            return random.randint(0, self.act_dim - 1)
        with torch.no_grad():
            return self.actor(obs).argmax(dim=-1).item()

    def compute_loss(self, obs, act, rew, next_obs, done, gamma, loss_fn):
        G_traj = []
        G = 0
        for r in rew[::-1]:
            G = r + gamma * G
            G_traj.insert(0, G)
        G_traj = torch.tensor(G_traj)

        pi, v = self.forward(obs, act)
