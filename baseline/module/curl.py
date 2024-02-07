import torch
import torch.nn as nn
import numpy as np
from skimage.util.shape import view_as_windows
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, RandomHorizontalFlip
import os

# Define the soft update for parameters
def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

# PixelEncoder class
class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, feature_dim, num_layers=3, num_filters=32, output_logits=False):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.output_logits = output_logits

        # Define layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2),
            nn.ReLU()
        )
        self.fc = nn.Linear(256 * 2 * 2, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.apply(weight_init)

    def forward_conv(self, obs):
        obs = obs / 255.
        x = self.conv_layers(obs)
        h = x.view(x.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)
        if detach:
            h = h.detach()
        h_fc = self.fc(h)
        h_norm = self.ln(h_fc)
        return torch.tanh(h_norm) if not self.output_logits else h_norm

# Define available encoders
_AVAILABLE_ENCODERS = {'pixel': PixelEncoder}

# Function to create encoder
def make_encoder(encoder_type, feature_dim, num_layers, num_filters, output_logits=False):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](feature_dim, num_layers, num_filters, output_logits)

# Weight initialization function
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.orthogonal_(m.weight.data[:, :, m.weight.size(2) // 2, m.weight.size(3) // 2])
        m.bias.data.fill_(0.0)

# Function for random cropping
def random_crop(imgs, output_size):
    n = imgs.shape[0]
    crop_max = imgs.shape[-1] - output_size
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    windows = view_as_windows(
        imgs, (1, output_size, output_size, 1))[..., 0,:,:, 0]
    return windows[np.arange(n), w1, h1]

# CURL Class
class CURL(nn.Module):
    def __init__(self, z_dim, batch_size, encoder, encoder_target, output_type="continuous"):
        super(CURL, self).__init__()
        self.batch_size = batch_size
        self.encoder = encoder
        self.encoder_target = encoder_target
        self.encoder_target.load_state_dict(self.encoder.state_dict())
        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        self.output_type = output_type

    def encode(self, x, detach=False, ema=False):
        with torch.no_grad() if ema else torch.enable_grad():
            z_out = self.encoder_target(x) if ema else self.encoder(x)
        return z_out.detach() if detach else z_out

    def compute_logits(self, z_a, z_pos):
        Wz = torch.matmul(self.W, z_pos.T)
        logits = torch.matmul(z_a, Wz)
        return logits - torch.max(logits, 1)[0][:, None]

# CurlAgent Class
class CurlAgent(object):
    def __init__(self, device, encoder_feature_dim=32, encoder_lr=1e-3, encoder_tau=0.005, num_layers=4, num_filters=32, curl_latent_dim=128):
        self.device = device
        self.encoder_tau = encoder_tau
        self.curl_latent_dim = curl_latent_dim

        # Initialize encoder and target encoder
        self.encoder = make_encoder('pixel', encoder_feature_dim, num_layers, num_filters, output_logits=True).to(device)
        self.encoder_target = make_encoder('pixel', encoder_feature_dim, num_layers, num_filters, output_logits=True).to(device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())

        # Initialize CURL
        self.CURL = CURL(encoder_feature_dim, self.curl_latent_dim, self.encoder, self.encoder_target, output_type='continuous').to(device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=encoder_lr)
        self.cpc_optimizer = torch.optim.Adam(self.CURL.parameters(), lr=encoder_lr)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.train()

    def train(self, training=True):
        self.training = training
        self.CURL.train(training)

    def update_cpc(self, obs_anchor, obs_pos, step, writer, training):
        z_a = self.CURL.encode(obs_anchor)
        z_pos = self.CURL.encode(obs_pos, ema=True)
        logits = self.CURL.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        loss = self.cross_entropy_loss(logits, labels)

        if training:
            self.encoder_optimizer.zero_grad()
            self.cpc_optimizer.zero_grad()
            loss.backward()
            self.encoder_optimizer.step()
            self.cpc_optimizer.step()
            writer.add_scalar('Loss curl', loss, step)
        return loss

    def update(self, obs, step, writer, training=True):
        obses = obs

        if step % 2 == 0:
            soft_update_params(self.encoder, self.encoder_target, self.encoder_tau)

        obs_anchor = random_crop(obses.cpu().numpy(), 64)
        pos = obses.cpu().numpy().copy()
        obs_pos = random_crop(pos, 64)
        obs_anchor = torch.as_tensor(obs_anchor, device=self.device).float()
        obs_pos = torch.as_tensor(obs_pos, device=self.device).float()
        return self.update_cpc(obs_anchor, obs_pos, step, writer, training)
    
    def encode(self, obs):
        obs = torch.as_tensor(obs, device=self.device).float()
        z_a = self.CURL.encode(obs, detach=True)
        return z_a

    
    def save_curl(self):
        torch.save(
            self.CURL.state_dict(), '../pretrained/curl.pt'
        )

    def load(self):
        self.CURL.load_state_dict(
            torch.load('../pretrained/curl.pt')
        )

# Function to make CurlAgent
def make_agent(device, encoder_feature_dim=32):
    return CurlAgent(
        device=device,
        encoder_feature_dim=32,
        encoder_lr=1e-3,
        encoder_tau=0.05,
        num_layers=4,
        num_filters=32,
        curl_latent_dim=128
    )