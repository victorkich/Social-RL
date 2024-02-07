import torch
import torch.nn as nn
import random

EPSILON = 0.2


class DQNController(nn.Module):
    def __init__(self, obs_dim: int = 256 + 32, hidden_dim: int = 128, act_dim: int = 5):
        super(DQNController, self).__init__()
        self.hidden_dim = hidden_dim
        self.act_dim = act_dim
        self.obs_dim = obs_dim

        self.layer = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )

    def forward(self, latent_z, hidden_state):
        x = torch.cat([latent_z, hidden_state], dim=-1)
        return self.layer(x)

    def sample(self, latent_z, hidden_state, epsilon=EPSILON):
        if random.random() < epsilon:
            return random.randint(0, self.act_dim - 1)
        with torch.no_grad():
            x = torch.cat([latent_z, hidden_state], dim=-1)
            return self.layer(x).argmax(dim=-1).item()


class A2CController(nn.Module):
    def __init__(self, obs_dim: int = 256 + 32, hidden_dim: int = 128, act_dim: int = 5):
        super(A2CController, self).__init__()
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

    def forward(self, latent_z, hidden_state, action):
        x = torch.cat([latent_z, hidden_state], dim=-1)
        y = torch.cat([latent_z, hidden_state, action], dim=-1)
        return self.actor(x), self.critic(y)

    def sample(self, latent_z, hidden_state, epsilon=EPSILON):
        if random.random() < epsilon:
            return random.randint(0, self.act_dim - 1)
        with torch.no_grad():
            x = torch.cat([latent_z, hidden_state], dim=-1)
            return self.actor(x).argmax(dim=-1).item()
