import numpy as np
import torch
from PIL import Image
from module.vae import VAE
from module.curl import make_agent
import os
import argparse
from tqdm import tqdm


def convert_npz2img(src: str = "../data/rollout", dst: str = "vae_dataset"):
    """convert npz to raw png images for training vae efficiently"""
    filenames = os.listdir(src)
    for i, file in tqdm(enumerate(filenames)):
        data = np.load(os.path.join(src, file))
        obs = data["obs"]
        for j in range(obs.shape[0]):
            img = Image.fromarray(obs[j])
            img.save(os.path.join(dst, f"{i}_{j}.png"))


def encode_dataset2latent(start_rollout: int, end_rollout: int, src: str, dst: str, type_encoder: str):
    """encode .npz rollouts to latent vectors for training rnn efficiently"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if type_encoder == 'vae':
        params = torch.load("pretrained/vae.pt")
        encoder = VAE(img_channels=3, latent_size=32)
        encoder.load_state_dict(params)
        encoder = encoder.to(device)
    else:
        encoder = make_agent(device=device, encoder_feature_dim=32)
        encoder.load()

    def encode_batch_obs(batch_obs: np.ndarray):
        batch_obs = torch.from_numpy(batch_obs).to(device)
        batch_obs = batch_obs.permute(0, 3, 1, 2)
        assert batch_obs.shape == (batch_obs.shape[0], 3, 64, 64)
        batch_obs = batch_obs.float()
        if type_encoder == 'vae':
            batch_z = encoder.encode(batch_obs)
        else:
            batch_z = encoder.encode(batch_obs)
        batch_z = batch_z.cpu().numpy()
        return batch_z

    for i in range(start_rollout, end_rollout):
        data = np.load(os.path.join(src, f"{i}.npz"))
        obs = data["states"]
        act = data["act"]
        next_obs = data["next_states"]

        latent_obs = encode_batch_obs(obs)
        latent_next_obs = encode_batch_obs(next_obs)
        np.savez(os.path.join(dst, f"{i}"), latent_obs=latent_obs,
                 act=act, latent_next_obs=latent_next_obs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess the collected rollouts")
    parser.add_argument("--raw2latent", action="store_true", help="convert raw frames npz to latent vector and store in a .npz file")
    parser.add_argument("--npz2img", action="store_true", help="convert npz to raw png images")
    args = parser.parse_args()

    if args.raw2latent:
        encode_dataset2latent()
    elif args.npz2img:
        convert_npz2img()
    else:
        raise ValueError("invalid argument")

# usage:
# python dataset_generate.py --raw2latent --start 0 --end 1000 --src dataset --dst vae_dataset
# python dataset_generate.py --npz2img --start 0 --end 200 --src dataset --dst vae_dataset
