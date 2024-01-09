import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
from torchvision.utils import save_image
import csv

# Definições de dimensões e hiperparâmetros
INPUT_DIM = (3, 64, 64)  # Canais x Altura x Largura (PyTorch usa canais primeiro)
Z_DIM = 32
BATCH_SIZE = 2048
LEARNING_RATE = 0.0001
KL_TOLERANCE = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Sampling(nn.Module):
    def forward(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.fc_mu = nn.Linear(128 * 4 * 4, Z_DIM)
        self.fc_log_var = nn.Linear(128 * 4 * 4, Z_DIM)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(Z_DIM, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 64, 6, stride=3, padding=2, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 64, 5, stride=3, padding=2, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 5, stride=3, padding=2, output_padding=0)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 5, stride=3, padding=2, output_padding=0)

    def forward(self, z):
        z = self.fc(z)
        z = z.view(z.size(0), 1024, 1, 1)
        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        z = F.relu(self.deconv3(z))
        reconstruction = torch.sigmoid(self.deconv4(z))
        return reconstruction

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.sampling = Sampling()
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

    def loss_function(self, recon_x, x, mu, log_var):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD

# Modificando a função train_step para retornar também a imagem reconstruída
def train_step(model, data):
    model.train()
    model.optimizer.zero_grad()
    recon_batch, mu, log_var = model(data)
    loss = model.loss_function(recon_batch, data, mu, log_var)
    loss.backward()
    model.optimizer.step()
    return loss.item(), recon_batch

# Função para salvar imagens de entrada e saída lado a lado
def save_reconstructed_images(recon_images, original_images, epoch, folder="results"):
    os.makedirs(folder, exist_ok=True)

    # Convertendo de BGR para RGB
    original_images_rgb = original_images[:10, [2, 1, 0], :, :]
    recon_images_rgb = recon_images[:10, [2, 1, 0], :, :]

    # Concatenando as imagens horizontalmente dentro de cada conjunto
    original_images_line = torch.cat([original_images_rgb[i].unsqueeze(0) for i in range(original_images_rgb.size(0))], dim=0)
    recon_images_line = torch.cat([recon_images_rgb[i].unsqueeze(0) for i in range(recon_images_rgb.size(0))], dim=0)

    # Concatenando os dois conjuntos de imagens verticalmente
    comparison = torch.cat((original_images_line, recon_images_line), dim=0)

    # Salvando a imagem - defina nrow para o número de imagens em cada linha
    save_image(comparison.cpu(), f'{folder}/reconstruction_{epoch}.png', nrow=10)

def train_vae(model, data_loader, epochs=10, save_interval=250):
    initial_epochs = epochs
    epoch = 0
    training_data = []  # Lista para armazenar dados de treinamento (época, perda)

    while epoch < initial_epochs:
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in tqdm(enumerate(data_loader)):
            data = data.to(device)
            loss, recon_batch = train_step(model, data)
            train_loss += loss

            # Salvar imagens a cada 'save_interval' épocas
            if epoch % save_interval == 0 and batch_idx == len(data_loader) - 1:
                save_reconstructed_images(recon_batch, data, epoch)

        epoch_loss = train_loss / len(data_loader)
        training_data.append((epoch, epoch_loss))  # Armazenar dados de treinamento
        epoch += 1
        print(f'Epoch: {epoch}/{initial_epochs}, Loss: {epoch_loss}')

        # Verificar se o usuário deseja modificar o número de épocas
        if epoch == initial_epochs:
            response = input("Continue training? Enter new number of epochs: ").strip()
            if response.lower() == 'n':
                break
            additional_epochs = int(response)
            initial_epochs = epoch + additional_epochs

    # Salvando os dados de treinamento em um arquivo CSV
    with open('training_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Loss'])
        writer.writerows(training_data)

    return model
