import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, RandomHorizontalFlip
import os
from module.curl import make_agent

# Define image transformation
transform = Compose([
    ToTensor(),
    RandomHorizontalFlip(p=0.5),
    Resize((76, 76))
])

# Function to transform image
def img_transform(frame: np.ndarray):
    frame = transform(Image.fromarray(frame))
    assert frame.shape == (3, 76, 76), f"frame shape is {frame.shape}"
    return frame

# CLDataset class
class CLDataset(Dataset):
    def __init__(self, root_dir="vae_dataset", meta_file="meta.json"):
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
    EPOCHS = 150
    print('Starting training...')
    dataset = CLDataset("/home/dranaju/vae_dataset")
    writer = SummaryWriter("log/curl")
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)
    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = make_agent(device, encoder_feature_dim=32)
    step = 0
    #agent.load()

    for epoch in range(EPOCHS):
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
        print(f'Evaluate')
        with torch.no_grad():
            for batch_raw in test_dataloader:
                batch_raw = batch_raw.to(device)
                loss = agent.update(batch_raw, step, writer, training=False)
                # step += 1
                losses.append(loss.item())
        writer.add_scalar("test/loss", np.mean(losses), epoch)
        # agent.save_curl()

if __name__ == "__main__":
    train_cl()
