import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from IPython import display
import os
import ipywidgets as widgets
from IPython.display import display
import torch.nn.functional as F

# Substitua pelo caminho correto do seu arquivo de definição do modelo VAE em PyTorch
from vae.arch_torch import VAE

np.set_printoptions(precision=4, suppress=True)

# Carregar o modelo PyTorch VAE
vae = VAE()
vae.load_state_dict(torch.load('./vae/weights.pth'))
vae.eval()  # Coloca o modelo em modo de avaliação

def preprocess_obs(obs, input_dim=(3, 64, 64)):
    # Certifique-se de que obs é um array NumPy
    if isinstance(obs, list) or obs.dtype == object:
        obs = np.array(obs.tolist(), dtype=np.float32)

    # Redimensionar a imagem para o tamanho de entrada esperado
    if obs.shape != (input_dim[1], input_dim[2], input_dim[0]):
        obs = resize(obs, (input_dim[1], input_dim[2]), anti_aliasing=True)

    # Reordenar para C, H, W para PyTorch
    obs = np.transpose(obs, (2, 0, 1))
    return obs

DIR_NAME = './data/rollout/'
file = os.listdir(DIR_NAME)[1]
print(os.listdir(DIR_NAME))
obs_data = np.load(DIR_NAME + file, allow_pickle=True)['obs']

# Exemplo de processamento de uma observação
obs = preprocess_obs(obs_data[0])
obs_tensor = torch.tensor([obs], dtype=torch.float32)  # Adiciona dimensão de batch

def visualize_reconstruction(obs, vae_model):
    obs_processed = preprocess_obs(obs)
    obs_tensor = torch.tensor([obs_processed], dtype=torch.float32)

    with torch.no_grad():
        #z_decoded = vae_model(obs_tensor)[0]
        #print(z_decoded)
        mu, log_var = vae.encoder(obs_tensor)
        z = vae.sampling(mu, log_var)
        z_decoded = vae.decoder(z)
        BCE = F.binary_cross_entropy(z_decoded, obs_tensor, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        print(BCE + KLD)
        print(z_decoded)
        z_decoded = np.transpose(z_decoded.cpu().numpy()[0], (1, 2, 0))

    obs_to_show = np.transpose(obs_processed, (1, 2, 0)) if obs_processed.ndim == 3 else obs

        # Normalizar se os valores forem maiores que 1
    if obs_to_show.max() > 1.0:
        obs_to_show /= 255.0
    if z_decoded.max() > 1.0:
        z_decoded /= 255.0

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(obs_to_show)
    plt.title("Original")
    plt.subplot(122)
    plt.imshow(z_decoded)
    plt.title("Reconstrução")
    plt.show()

# Exemplo de uso
visualize_reconstruction(obs_data[0], vae)

def decode_latent(vae, z):
    with torch.no_grad():  # Garantir que não estamos calculando gradientes
        z_tensor = torch.tensor([z], dtype=torch.float32)
        decoded_img = vae.decoder(z_tensor)
        decoded_img = np.transpose(decoded_img.cpu().numpy()[0], (1, 2, 0))  # Transformar para formato HWC para visualização
    return decoded_img

def interactive_plot(vae):
    def update_plot(*z_vals):
        z = np.array(z_vals, dtype=np.float32)
        decoded_img = decode_latent(vae, z)
        plt.imshow(decoded_img)
        plt.show()

    sliders = []
    for i in range(vae.z_dim):  # Assumindo que z_dim é a dimensão do espaço latente
        slider = widgets.FloatSlider(value=0.0, min=-3.0, max=3.0, step=0.1, description=f'z[{i}]')
        sliders.append(slider)

    interact(update_plot, **{slider.description: slider for slider in sliders})

# Chame esta função para iniciar a visualização interativa
interactive_plot(vae)
