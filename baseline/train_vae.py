from module.vae import VAE
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


transform = v2.Compose([
    v2.ToImage(),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32),
])


def img_transform(frame: np.ndarray):
    frame = transform(frame)
    assert frame.shape == (3, 64, 64), f"frame shape is {frame.shape}"
    return frame


class VAEDataset(Dataset):
    def __init__(self, root_dir: str = "vae_dataset", meta_file: str = "meta.json"):
        self.meta_file = meta_file
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


def train_vae():
    EPOCH = 150
    min_loss = np.inf
    dataset = VAEDataset("vae_dataset", "meta.json")
    writer = SummaryWriter("log/vae2")
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)
    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vae = VAE(img_channels=3, latent_size=32).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    # training pipeline
    for epoch in range(EPOCH):
        losses = []
        vae.train()
        for batch_raw in train_dataloader:
            batch_raw = batch_raw.to(device)
            reconstruction, mu, logsigma = vae(batch_raw)
            optimizer.zero_grad()
            loss = loss_fn(reconstruction, batch_raw)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        writer.add_scalar("train/loss", np.mean(losses), epoch)

        losses = []
        vae.eval()
        record = True
        for batch_raw in test_dataloader:
            batch_raw = batch_raw.to(device)
            reconstruction, mu, logsigma = vae(batch_raw)
            loss = loss_fn(reconstruction, batch_raw)
            losses.append(loss.item())
            if record:
                record = False
                record_idx = random.randint(0, len(batch_raw)-1)
                writer.add_image(
                    "train/raw", batch_raw[record_idx].detach().cpu().to(torch.uint8), epoch)
                writer.add_image(
                    "train/reconstruction", reconstruction[record_idx].detach().cpu().to(torch.uint8), epoch)
        if min_loss > np.mean(losses):
            min_loss = np.mean(losses)
            torch.save(vae.state_dict(), "pretrained/vae2.pt")
        writer.add_scalar("test/loss", np.mean(losses), epoch)


if __name__ == "__main__":
    train_vae()
