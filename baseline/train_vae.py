from module.vanilla_vae import VanillaVAE as VAE
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
import random
import os
from PIL import Image
import torch.nn as nn
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
import torchvision
torchvision.disable_beta_transforms_warning()

transform = v2.Compose([
    v2.ToDtype(torch.float),
    v2.RandomHorizontalFlip(p=0.5),
])

def img_transform(frame: np.ndarray):
    # Converter a imagem PIL para um tensor do PyTorch
    frame = to_tensor(frame)
    # Aplicar as transformações
    frame = transform(frame)
    # Verificação de forma pode ser mantida se a transformação resultar na forma esperada
    assert frame.shape == (3, 64, 64), f"frame shape is {frame.shape}"
    return frame

class VAEDataset(Dataset):
    def __init__(self, root_dir: str = "vae_dataset"):
        self.root_dir = root_dir
        self.filenames = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root_dir, self.filenames[idx]))
        raw = img_transform(img)
        return raw

class VAERolloutDataset(Dataset):
    def __init__(self, root_dir: str = "dataset", meta_file: str = "meta.json", episode_len: int = 200):
        self.meta_file = meta_file
        self.root_dir = root_dir
        self.filenames = os.listdir(self.root_dir)
        self.episode_len = episode_len

    def __len__(self):
        return self.episode_len

    def __getitem__(self, index):
        data = np.load(os.path.join(self.root_dir, f"{index}.npz"))
        frames = data["states"].astype(np.uint8)
        frames = frames.transpose(0, 3, 1, 2)
        raw = img_transform(raw)
        return raw

def train_vae():
    EPOCH = 4000
    batch_size = 1024
    latent_dim = 32
    dataset = VAEDataset("vae_dataset")
    min_loss = np.inf
    writer = SummaryWriter("log/vae2")
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)
    # Calcula M_N aqui, após a divisão do dataset
    M_N = 1 / len(train_dataset)  # Use len(train_dataset) para o cálculo correto após a divisão
    print("M_N:", M_N)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vae = VAE(in_channels=3, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    
    # training pipeline
    print("---- TRAINING VAE ----")
    for epoch in tqdm(range(EPOCH)):
        losses = []
        recons_losses = []
        kld_losses = []
        vae.train()
        for batch_raw in train_dataloader:
            batch_raw = batch_raw.to(device)
            reconstruction, input, mu, logsigma = vae(batch_raw)
            optimizer.zero_grad()
            loss_dict = vae.loss_function(reconstruction, input, mu, logsigma, M_N=M_N)
            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            recons_losses.append(loss_dict['Reconstruction_Loss'].cpu().item())
            kld_losses.append(loss_dict['KLD'].cpu().item())
        writer.add_scalar("train/loss", np.mean(losses), epoch)
        writer.add_scalar("train/reconstruction_loss", np.mean(recons_losses), epoch)
        writer.add_scalar("train/kld_loss", np.mean(kld_losses), epoch)

        losses = []
        recons_losses = []
        kld_losses = []
        vae.eval()
        record = True
        for batch_raw in test_dataloader:
            batch_raw = batch_raw.to(device)
            reconstruction, input, mu, logsigma = vae(batch_raw)
            # Chame a função de perda com M_N como um argumento
            loss_dict = vae.loss_function(reconstruction, input, mu, logsigma, M_N=M_N)
            loss = loss_dict['loss']
            losses.append(loss.item())
            recons_losses.append(loss_dict['Reconstruction_Loss'].cpu().item())
            kld_losses.append(loss_dict['KLD'].cpu().item())
            if record:
                record = False
                record_idx = random.randint(0, len(batch_raw)-1)
                writer.add_image("train/raw", (batch_raw[record_idx].detach().cpu().float() * 255).to(torch.uint8), epoch)
                writer.add_image("train/reconstruction", (reconstruction[record_idx].detach().cpu().float() * 255).to(torch.uint8), epoch)

        if min_loss > np.mean(losses):
            min_loss = np.mean(losses)
            torch.save(vae.state_dict(), "../pretrained/vae2.pt")
        writer.add_scalar("test/loss", np.mean(losses), epoch)
        writer.add_scalar("test/reconstruction_loss", np.mean(recons_losses), epoch)
        writer.add_scalar("test/kld_loss", np.mean(kld_losses), epoch)


if __name__ == "__main__":
    train_vae()
