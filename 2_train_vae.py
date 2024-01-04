from vae.arch import VAE
import argparse
import numpy as np
from skimage.transform import resize
import os
from tqdm import tqdm

DIR_NAME = './data/rollout/'

SCREEN_SIZE_X = 64
SCREEN_SIZE_Y = 64

def import_data(N, M):
  filelist = os.listdir(DIR_NAME)
  filelist = [x for x in filelist if x != '.DS_Store']
  filelist.sort()
  length_filelist = len(filelist)

  if length_filelist > N:
    filelist = filelist[:N]

  if length_filelist < N:
    N = length_filelist

  data = np.zeros((100 * N, 64, 64, 3), dtype=np.float32)  # Ajuste aqui para 100*N
  idx = 0
  file_count = 0

  for file in tqdm(filelist):
    print("File:", file)
    try:
      new_data = np.load(DIR_NAME + file, allow_pickle=True)['obs']
      if new_data.dtype == object:
        new_data = np.stack(new_data.tolist())
      new_data_resized = np.array([resize(img, (64, 64, 3)) for img in new_data])  # Redimensiona as imagens
      data[idx:(idx + new_data_resized.shape[0]), :, :, :] = new_data_resized

      idx += new_data_resized.shape[0]  # Atualize idx baseado no número de dados carregados
      file_count += 1

      if file_count % 50 == 0:
        print('Imported {} / {} ::: Current data size = {} observations'.format(file_count, N, idx))
    except Exception as e:
      print(e)
      print('Skipped {}...'.format(file))

  print('Imported {} / {} ::: Current data size = {} observations'.format(file_count, N, idx))
  return data, N

def main(args):
  new_model = args.new_model
  N = int(args.N)
  M = int(args.time_steps)
  epochs = int(args.epochs)

  vae = VAE()

  if not new_model:
    try:
      vae.set_weights('./vae/weights.h5')
    except:
      print("Either set --new_model or ensure ./vae/weights.h5 exists")
      raise

  try:
    data, N = import_data(N, M)
  except:
    print('NO DATA FOUND')
    raise
      
  print('DATA SHAPE = {}'.format(data.shape))

  for epoch in range(epochs):
    print('EPOCH ' + str(epoch))
    vae.save_weights('./vae/weights.h5')
    vae.train(data)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=('Train VAE'))
  parser.add_argument('--N',default = 10000, help='number of episodes to use to train')
  parser.add_argument('--new_model', action='store_true', help='start a new model from scratch?')
  parser.add_argument('--time_steps', type=int, default=300,
                        help='how many timesteps at start of episode?')
  parser.add_argument('--epochs', default = 1000, help='number of epochs to train for')
  args = parser.parse_args()

  main(args)
