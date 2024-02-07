import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class MDNRNN(nn.Module):
    def __init__(self, n_hidden, n_gaussians, latent_dim, action_dim):
        super(MDNRNN, self).__init__()
        self.n_hidden = n_hidden
        self.n_gaussians = n_gaussians
        self.latent_dim = latent_dim
        self.act_dim = action_dim

        self.lstm = nn.LSTM(latent_dim + action_dim, n_hidden)
        self.mdn = nn.Linear(n_hidden, 2 * latent_dim * n_gaussians)

    def init_hidden(self):
        h = torch.zeros(1, self.n_hidden)
        c = torch.zeros(1, self.n_hidden)
        return h, c

    def forward(self, latent_z, actions):
        """
        latent_z: (batch_size, seq_len, latent_dim)
        actions: (batch_size, seq_len, 1)
        """

        batch_size, seq_len = latent_z.shape[0], latent_z.shape[1]
        x = torch.cat([latent_z, actions], dim=-1)
        hidden, _ = self.lstm(x)
        assert hidden.shape == (batch_size, seq_len, self.n_hidden)
        prediction = self.mdn(hidden)
        mu = prediction[:, :, :self.n_gaussians * self.latent_dim].view(
            batch_size, seq_len, self.latent_dim, self.n_gaussians)
        log_sigma = prediction[:, :, self.n_gaussians * self.latent_dim:].view(
            batch_size, seq_len, self.latent_dim, self.n_gaussians)
        pred_z = torch.randn_like(mu) * torch.exp(log_sigma) + mu
        pred_z = torch.sum(pred_z, dim=-1)
        assert pred_z.shape == (
            batch_size, seq_len, self.latent_dim), f"pred_z shape is {pred_z.shape}"
        return pred_z

    def sample(self, latent_z, actions, h, c):
        """
        latent_z: (batch_size, latent_dim)
        actions: (batch_size, 1)
        """
        with torch.no_grad():
            x = torch.cat([latent_z, actions], dim=-1)
            assert x.shape == (x.shape[0], self.latent_dim + self.act_dim)
            hidden, (h, c) = self.lstm(x, (h, c))
            prediction = self.mdn(hidden)
            mu = prediction[:, :self.n_gaussians * self.latent_dim].view(
                -1, self.latent_dim, self.n_gaussians)
            sigma = prediction[:, self.n_gaussians * self.latent_dim:].view(
                -1, self.latent_dim, self.n_gaussians)
            pred_z = torch.randn_like(mu) * torch.exp(sigma) + mu
            pred_z = torch.sum(pred_z, dim=-1)
            assert pred_z.shape == (pred_z.shape[0],
                                    self.latent_dim), f"pred_z shape is {pred_z.shape}"
            return pred_z, h, c
