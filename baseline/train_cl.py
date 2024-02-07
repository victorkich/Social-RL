import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from PIL import Image
import os
from module.curl import make_agent
from tqdm import tqdm
from torchvision.transforms import v2

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
    EPOCHS = 2000
    batch_size = 1024
    dataset = CLDataset("vae_dataset")
    writer = SummaryWriter("log/curl")
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = make_agent(device, encoder_feature_dim=32)
    step = 0

    # training pipeline
    print("---- TRAINING CL ----")
    for epoch in tqdm(range(EPOCHS)):
        agent.CURL.train()
        losses = []
        print(f'Epoch {epoch}')
        for batch_raw in train_dataloader:
            batch_raw = batch_raw.to(device)
            loss = agent.update(batch_raw, step, writer)
            step += 1
            losses.append(loss.item())
        writer.add_scalar("train/loss", np.mean(losses), epoch)
        agent.save_curl()

        agent.CURL.eval()
        losses = []
        with torch.no_grad():
            for batch_raw in test_dataloader:
                batch_raw = batch_raw.to(device)
                loss = agent.update(batch_raw, step, writer, training=False)
                losses.append(loss.item())
        writer.add_scalar("test/loss", np.mean(losses), epoch)


if __name__ == "__main__":
    train_cl()
