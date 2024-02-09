import torch
import torch.nn as nn
import numpy as np
from skimage.util.shape import view_as_windows
import torch.nn.functional as F
from torchvision.transforms import v2

transform = v2.Compose([
    v2.Resize((64, 64), antialias=True),
    v2.ToDtype(torch.float)
])

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

class PixelDecoder(nn.Module):
    """Convolutional decoder for pixel observations."""
    def __init__(self, feature_dim, output_channels=3, num_layers=4, num_filters=32):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        # Começa com uma camada linear para levar o espaço latente de volta ao tamanho intermediário
        self.fc = nn.Linear(feature_dim, 256 * 2 * 2)

        # Define as camadas deconvolucionais
        modules = []
        hidden_dims = [256, 128, 64, 32][:num_layers]
        hidden_dims.reverse()  # Começa com a dimensão maior para a menor

        # Ajuste: A primeira camada deconvolucional deve começar com 256 canais de entrada
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(256, hidden_dims[0], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dims[0]),
                nn.ReLU())
        )
        
        for i in range(1, len(hidden_dims)):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i-1], hidden_dims[i], kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dims[i]),
                    nn.ReLU())
            )
        
        self.decoder_layers = nn.Sequential(*modules)

        # Última camada para mapear para os canais de saída
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Tanh para mapear os valores de saída para [0, 1]
        )

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(-1, 256, 2, 2)  # Redimensiona para começar a deconvolução
        x = self.decoder_layers(x)
        x = self.final_layer(x)
        return x

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
    windows = view_as_windows(imgs, (1, output_size, output_size, 1))[..., 0,:,:, 0]
    return windows[np.arange(n), w1, h1]

# CURL Class
class CURL(nn.Module):
    def __init__(self, z_dim, batch_size, encoder, encoder_target, decoder, output_type="continuous"):
        super(CURL, self).__init__()
        self.batch_size = batch_size
        self.encoder = encoder
        self.encoder_target = encoder_target
        self.decoder = decoder  # Adiciona o decodificador
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
    
    def decode(self, z):
        return self.decoder(z)

# CurlAgent Class
class CurlAgent(object):
    def __init__(self, device, encoder_feature_dim=32, encoder_lr=1e-3, encoder_tau=0.005, num_layers=4, num_filters=32, curl_latent_dim=128):
        self.device = device
        self.encoder_tau = encoder_tau
        self.curl_latent_dim = curl_latent_dim

        # Inicializa o codificador e o codificador alvo
        self.encoder = make_encoder('pixel', encoder_feature_dim, num_layers, num_filters, output_logits=True).to(device)
        self.encoder_target = make_encoder('pixel', encoder_feature_dim, num_layers, num_filters, output_logits=True).to(device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())

        # Inicializa o decodificador
        self.decoder = PixelDecoder(feature_dim=encoder_feature_dim, output_channels=3, num_layers=num_layers, num_filters=num_filters).to(device)

        # Inicializa CURL com o decodificador
        self.CURL = CURL(encoder_feature_dim, self.curl_latent_dim, self.encoder, self.encoder_target, self.decoder, output_type='continuous').to(device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=encoder_lr)
        self.cpc_optimizer = torch.optim.Adam(self.CURL.parameters(), lr=encoder_lr)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.train()

    def train(self, training=True):
        self.training = training
        self.CURL.train(training)

    def update(self, obs, training=True):
        obses = obs  # obs é a imagem original antes da augmentação

        # Realiza as augmentações necessárias para o aprendizado contrastivo
        obs_anchor = random_crop(obses.cpu().numpy(), 64)
        pos = obses.cpu().numpy().copy()
        obs_pos = random_crop(pos, 64)
        obs_anchor = torch.as_tensor(obs_anchor, device=self.device).float()
        obs_pos = torch.as_tensor(obs_pos, device=self.device).float()

        # Codifica as imagens para o espaço latente
        z_a = self.CURL.encode(obs_anchor)
        z_pos = self.CURL.encode(obs_pos, ema=True)

        # Calcula a perda contrastiva
        logits = self.CURL.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        contrastive_loss = self.cross_entropy_loss(logits, labels)

        # Decodifica a partir do espaço latente para a reconstrução
        reconstructed_anchor = self.CURL.decode(z_a)
        reconstructed_pos = self.CURL.decode(z_pos)

        # Prepara a imagem original para a comparação
        obses_transformed = transform(obses)
        recon_loss_anchor = F.mse_loss(reconstructed_anchor, obses_transformed.to(self.device))
        recon_loss_pos = F.mse_loss(reconstructed_pos, obses_transformed.to(self.device))

        # Calcula a perda total e faz a retropropagação
        total_loss = contrastive_loss + recon_loss_anchor + recon_loss_pos
        if training:
            self.encoder_optimizer.zero_grad()
            self.cpc_optimizer.zero_grad()
            total_loss.backward()
            self.encoder_optimizer.step()
            self.cpc_optimizer.step()

            # Aplica o soft update nos parâmetros do encoder_target após a atualização dos parâmetros do encoder
            soft_update_params(self.encoder, self.encoder_target, self.encoder_tau)

        return {'loss': total_loss.item(), 'contrastive_loss': contrastive_loss.item(), 'recon_anchor_loss': recon_loss_anchor.item(), 'recon_pos_loss': recon_loss_pos.item()}

    def encode(self, obs, ema=False):
        obs = torch.as_tensor(obs, device=self.device).float()
        z_a = self.CURL.encode(obs, detach=True, ema=ema)
        return z_a
    
    def sample(self, obs, anchor=False, pos=False):
        obses = obs
        obs_anchor = None
        reconstructed_anchor = None
        obs_pos = None
        reconstructed_pos = None

        if anchor:
            obs_anchor = random_crop(obses.cpu().numpy(), 64)
            z_anchor = self.encode(obs_anchor)  # Codifica a imagem anchor
            reconstructed_anchor = self.CURL.decode(z_anchor)  # Decodifica a partir do espaço latente
        if pos:
            pos = obses.cpu().numpy().copy()
            obs_pos = random_crop(pos, 64)
            z_pos = self.encode(obs_pos, ema=True)  # Codifica a imagem pos
            reconstructed_pos = self.CURL.decode(z_pos)

        obses = transform(obses)
        raw_z = self.encode(obs)
        recon_raw = self.CURL.decode(raw_z)

        return {'obs': obses, 'recon_raw':recon_raw, 'obs_anchor':obs_anchor, 
                'recon_anchor':reconstructed_anchor, 'obs_pos':obs_pos, 'recon_pos':reconstructed_pos}

    def save_curl(self):
        torch.save(self.CURL.state_dict(), '../pretrained/curl2.pt')

    def load(self):
        self.CURL.load_state_dict(torch.load('../pretrained/curl2.pt', map_location=self.device))

# Function to make CurlAgent
def make_agent(device, encoder_feature_dim=32):
    return CurlAgent(
        device=device,
        encoder_feature_dim=encoder_feature_dim,
        encoder_lr=1e-3,
        encoder_tau=0.05,
        num_layers=4,
        num_filters=32,
        curl_latent_dim=128
    )
