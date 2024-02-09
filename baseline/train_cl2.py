import torch
import torchvision
torchvision.disable_beta_transforms_warning()
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from PIL import Image
import os
from module.curl2 import make_agent
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
from torchvision.transforms import v2
import random

transform = v2.Compose([
    v2.Resize((76, 76), antialias=True),
    v2.ToDtype(torch.float),
    v2.RandomHorizontalFlip(p=0.5),
])

def img_transform(frame: np.ndarray):
    # Converter a imagem PIL para um tensor do PyTorch
    frame = to_tensor(frame)
    # Aplicar as transformações
    frame = transform(frame)
    # Verificação de forma pode ser mantida se a transformação resultar na forma esperada
    assert frame.shape == (3, 76, 76), f"frame shape is {frame.shape}"
    return frame

# CLDataset class
class CLDataset(Dataset):
    def __init__(self, root_dir="vae_dataset"):
        self.root_dir = root_dir
        self.filenames = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.filenames[idx])
        img = Image.open(img_path)
        return img_transform(np.array(img))

# Main training function
def train_cl():
    EPOCHS = 10000
    batch_size = 1024
    latent_dim = 32
    dataset = CLDataset("vae_dataset")
    writer = SummaryWriter("log/curl2")
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = make_agent(device, encoder_feature_dim=latent_dim)
    step = 0

    # training pipeline
    print("---- TRAINING CL2 ----")
    for epoch in tqdm(range(EPOCHS)):
        agent.CURL.train()
        losses = []
        recon_anchor_losses = []
        recon_pos_losses = []
        contrastive_losses = []
        for batch_raw in train_dataloader:
            batch_raw = batch_raw.to(device)
            loss_dict = agent.update(batch_raw)
            step += 1
            losses.append(loss_dict['loss'])
            contrastive_losses.append(loss_dict['contrastive_loss'])
            recon_anchor_losses.append(loss_dict['recon_anchor_loss'])
            recon_pos_losses.append(loss_dict['recon_pos_loss'])

        writer.add_scalar("train/total_loss", np.mean(losses), epoch)
        writer.add_scalar("train/contrastive_loss", np.mean(contrastive_losses), epoch)
        writer.add_scalar("train/recon_anchor_loss", np.mean(recon_anchor_losses), epoch)
        writer.add_scalar("train/recon_pos_loss", np.mean(recon_pos_losses), epoch)
        agent.save_curl()

        agent.CURL.eval()
        record = True
        losses = []
        recon_anchor_losses = []
        recon_pos_losses = []
        contrastive_losses = []
        with torch.no_grad():
            for batch_raw in test_dataloader:
                batch_raw = batch_raw.to(device)
                loss_dict = agent.update(batch_raw, training=False)
                losses.append(loss_dict['loss'])
                contrastive_losses.append(loss_dict['contrastive_loss'])
                recon_anchor_losses.append(loss_dict['recon_anchor_loss'])
                recon_pos_losses.append(loss_dict['recon_pos_loss'])

                if record:
                    record = False
                    record_idx = random.randint(0, len(batch_raw)-1)
                    obs_dict = agent.sample(batch_raw[record_idx].unsqueeze(0), anchor=True, pos=True)
                    
                    # Ensure to squeeze the batch dimension and multiply by 255 to convert from [0, 1] to [0, 255]
                    writer.add_image("test/raw", obs_dict['obs'].squeeze(0), epoch, dataformats='CHW')
                    writer.add_image("test/aug_anchor", obs_dict['obs_anchor'].squeeze(0), epoch, dataformats='CHW')
                    writer.add_image("test/aug_pos", obs_dict['obs_pos'].squeeze(0), epoch, dataformats='CHW')
                    writer.add_image("test/recon_raw", obs_dict['recon_raw'].squeeze(0), epoch, dataformats='CHW')
                    writer.add_image("test/recon_anchor", obs_dict['recon_anchor'].squeeze(0), epoch, dataformats='CHW')
                    writer.add_image("test/recon_pos", obs_dict['recon_pos'].squeeze(0), epoch, dataformats='CHW')

        writer.add_scalar("test/total_loss", np.mean(losses), epoch)
        writer.add_scalar("test/contrastive_loss", np.mean(contrastive_losses), epoch)
        writer.add_scalar("test/recon_anchor_loss", np.mean(recon_anchor_losses), epoch)
        writer.add_scalar("test/recon_pos_loss", np.mean(recon_pos_losses), epoch)

if __name__ == "__main__":
    train_cl()
