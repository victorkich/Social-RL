import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from sklearn.model_selection import train_test_split
from module.mdnrnn import MDNRNN
from torch.utils.tensorboard import SummaryWriter


class RNNDataset(Dataset):
    def __init__(self, root_dir: str = "rnn_dataset", meta_file: str = "meta.json"):
        self.meta_file = meta_file
        self.root_dir = root_dir
        self.filenames = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        data = np.load(os.path.join(
            self.root_dir, self.filenames[idx]))
        latent_obs = data["latent_obs"]
        act = data["act"]
        latent_next_obs = data["latent_next_obs"]
        latent_obs = torch.from_numpy(latent_obs).float()
        act = torch.from_numpy(act).float()
        latent_next_obs = torch.from_numpy(latent_next_obs).float()

        return latent_obs, act, latent_next_obs


def train_mdnrnn():
    EPOCH = 100
    min_loss = np.inf
    dataset = RNNDataset()
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    writer = SummaryWriter("log/mdnrnn")

    mdnrnn = MDNRNN(n_hidden=256, n_gaussians=5,
                    latent_dim=32, action_dim=1).to(device)
    optimzer = torch.optim.Adam(mdnrnn.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for i in range(EPOCH):
        mdnrnn.train()
        losses = []
        for latent_obs, act, latent_next_obs in train_dataloader:
            latent_obs = latent_obs.to(device)
            act = act.to(device)
            latent_next_obs = latent_next_obs.to(device)

            pred_next_obs = mdnrnn(latent_obs, act.unsqueeze(-1))
            optimzer.zero_grad()
            loss = loss_fn(pred_next_obs, latent_next_obs)
            loss.backward()
            optimzer.step()
            losses.append(loss.item())
        writer.add_scalar("train_loss", np.mean(losses), i)

        mdnrnn.eval()
        losses = []
        for latent_obs, act, latent_next_obs in test_dataloader:
            latent_obs = latent_obs.to(device)
            act = act.to(device)
            latent_next_obs = latent_next_obs.to(device)
            pred_next_obs = mdnrnn(latent_obs, act.unsqueeze(-1))
            loss = loss_fn(pred_next_obs, latent_next_obs)
            losses.append(loss.item())
        writer.add_scalar("test_loss", np.mean(losses), i)
        if np.mean(losses) < min_loss:
            min_loss = np.mean(losses)
            torch.save(mdnrnn.state_dict(), "pretrained/mdnrnn.pt")


if __name__ == "__main__":
    train_mdnrnn()
