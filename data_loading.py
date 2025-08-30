import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import utils
import random

A_MIN = 0.0
A_MAX = 180.0
A_STEP = 15.0

E_MIN = 5.0
E_MAX = 15.0
E_STEP = 0.05

# how many times each image is duplicated and augmented
IMG_DUP = 10

def global_grid():

    A_axis = np.arange(A_MIN, A_MAX, A_STEP)
    E_axis = np.arange(E_MIN, E_MAX, E_STEP)

    return A_axis, E_axis

def place_image_on_grid(A_vals, E_vals, cx_vals):

    A_vals = np.array(A_vals)
    E_vals = np.array(E_vals)
    cx_vals = np.array(cx_vals)

    A_axis, E_axis = global_grid()
    num_A = len(A_axis)
    num_E = len(E_axis)

    image = torch.zeros((1, num_A, num_E))

    # get positions of nearest coordinates on grid
    A_idx = np.abs(A_vals[:, None] - A_axis[None, :]).argmin(axis=1)
    E_idx = np.abs(E_vals[:, None] - E_axis[None, :]).argmin(axis=1)

    for i in range(len(cx_vals)):
        ai, ei = A_idx[i], E_idx[i]
        image[0, ai, ei] = cx_vals[i]
        # image[1, ai, ei] = 1.0

    return image

def get_images(train_path, log=True, crop_coef=3, angle_p=0.25):

    with open(train_path, 'r') as f:
        data = json.load(f)

    n = len(data)
    images = []

    for i in range(n):

        points = data[i]['observable_sets'][0]['points']
        
        A_vals = []
        E_vals = []
        cx_vals = []

        for p in points:
            A_vals.append(p['theta_cm_out'])
            E_vals.append(p['cn_ex'])

            if log == True:
                cx = np.log10(p['dsdO'])
                cx_vals.append(cx)
            else:
                cx_vals.append(p['dsdO'])

        image = place_image_on_grid(A_vals, E_vals, cx_vals)

        for i in range(IMG_DUP):
            img = image.detach().clone()
            img = utils.random_crop(img, crop_coef, angle_p)
            images.append(img)

    return torch.stack(images, dim=0)

# get all targets from a training set
# returns a tensor of shape [n_samples, 2]
# energy, total width
def get_targets(train_path):

    with open(train_path, 'r') as f:
        data = json.load(f)

    n = len(data)
    tensors = []
    for i in range(n):

        levels = data[i]['levels'][0]
        energy = levels['energy']
        gamma_total = levels['Gamma_total']

        # normalise energy
        points = data[i]['observable_sets'][0]['points']
        df = pd.DataFrame(points)
        energies_abs = sorted(df['cn_ex'].unique())
        e_min, e_max = min(energies_abs), max(energies_abs)
        Er_unit = (energy - e_min) / max(e_max - e_min, 1e-8)
        Er_unit = float(np.clip(Er_unit, 0.0, 1.0))

        log10_gamma = float(np.log10(gamma_total + 1e-8))

        for i in range(IMG_DUP):
            tensors.append(torch.tensor([Er_unit, log10_gamma], dtype=torch.float32))

    return torch.stack(tensors, dim=0)

# input dataset for ResonanceCNN
class ResonanceDataset(Dataset):

    def __init__(self, images, targets, gradients=True):

        self.images = images # shape: [n_samples, 2, n_A, n_E]
        self.targets = targets # shape: [n_samples, 2]
        self.gradients = gradients

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        image = self.images[idx]
        target = self.targets[idx]

        if self.gradients:
            image = utils.sobel(image)

        return image, target