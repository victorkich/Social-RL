from typing import Any
import torch
import numpy as np
from PIL import Image


def image_array2tensor(array: np.ndarray):
    """convert a image (H, W, C) array to a tensor (C, H, W)"""
    array = array.transpose(2, 0, 1)
    img_tensor = torch.from_numpy(array).to(torch.float32)
    return img_tensor


def crop_frame(frame: np.ndarray) -> np.ndarray:
    """crop the frame to (64, 64, 3) np.ndarray"""
    img = Image.fromarray(frame)
    img = img.crop((0, 0, 96, 80)).resize((64, 64))
    img = np.array(img)
    assert img.shape == (64, 64, 3), f"img.shape = {img.shape}"
    return img


class Scheduler(object):
    def __init__(self, init_value: float, end_value: float, nsteps: int):
        self.init_value = init_value
        self.end_value = end_value
        self.nsteps = nsteps
        self.steps = 0

    def step(self):
        self.steps = min(self.steps + 1, self.nsteps)
        return self.end_value + (self.init_value - self.end_value) * (1 - self.steps / self.nsteps) ** 2

    def __call__(self):
        return self.step()
