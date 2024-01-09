from vae.arch_torch import VAE, train_vae  # Ajuste este importe conforme necessário
import argparse
import numpy as np
from skimage.transform import resize
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset

from matplotlib import pyplot as plt

DIR_NAME = './data/rollout/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCREEN_SIZE_X = 64
SCREEN_SIZE_Y = 64
BATCH_SIZE = 2048

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

from concurrent.futures import ProcessPoolExecutor

# Função para processar um único arquivo
def process_file(file):
    try:
        new_data = np.load(DIR_NAME + file, allow_pickle=True)['obs']
        obs = [preprocess_obs(d) for d in new_data]
        return obs
    except Exception as e:
        print(e)
        print('Skipped {}...'.format(file))
        return []

def import_data(N, M):
    filelist = os.listdir(DIR_NAME)
    filelist = [x for x in filelist if x != '.DS_Store']
    filelist.sort()
    length_filelist = len(filelist)

    if length_filelist > N:
        filelist = filelist[:N]

    if length_filelist < N:
        N = length_filelist

    data = []
    with ProcessPoolExecutor() as executor:
        # Utilizando tqdm para monitorar o progresso
        futures = [executor.submit(process_file, file) for file in filelist]
        for future in tqdm(futures, total=len(filelist), desc="Loading Files"):
            data.extend(future.result())

    data = np.array(data, dtype=np.float32)
    data /= 255.0  # Normalizando
    return torch.tensor(data, dtype=torch.float32).to(device), N

def main(args):
    new_model = args.new_model
    N = int(args.N)
    M = int(args.time_steps)
    epochs = int(args.epochs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = VAE().to(device)

    if not new_model:
        try:
            vae.load_state_dict(torch.load('./vae/weights.pth'))
        except:
            print("Either set --new_model or ensure ./vae/weights.pth exists")
            raise

    try:
        data, N = import_data(N, M)
    except:
        print('NO DATA FOUND')
        raise

    dataset = TensorDataset(data, data)  # PyTorch espera dados e alvos, então fornecemos data como ambos
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print('DATA SHAPE = {}'.format(data.shape))
    trained_vae = train_vae(vae, data_loader=dataloader, epochs=epochs)
    torch.save(trained_vae.state_dict(), './vae/weights.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train VAE'))
    parser.add_argument('--N', default = 10000, help='number of episodes to use to train')
    parser.add_argument('--new_model', action='store_true', help='start a new model from scratch?')
    parser.add_argument('--time_steps', type=int, default=300,
                            help='how many timesteps at start of episode?')
    parser.add_argument('--epochs', default = 1000, help='number of epochs to train for')
    args = parser.parse_args()

    main(args)
