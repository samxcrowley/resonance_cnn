import json
import gzip
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import utils

global_E_min = 8.0
global_E_max = 20.0
global_E_step = 0.1

global_A_min = 25.0
global_A_max = 175.0
global_A_step = 10.0

IMG_DUP = 10

def global_grid():

    E_axis = np.arange(global_E_min, global_E_max, global_E_step)
    A_axis = np.arange(global_A_min, global_A_max, global_A_step)

    return E_axis, A_axis

def place_image_on_grid(E_vals, A_vals, cx_vals):

    E_vals = np.array(E_vals)
    A_vals = np.array(A_vals)
    cx_vals = np.array(cx_vals)

    E_axis, A_axis = global_grid()
    num_E = len(E_axis)
    num_A = len(A_axis)

    image = torch.zeros((2, num_E, num_A))

    # get positions of nearest coordinates on grid
    E_idx = np.abs(E_vals[:, None] - E_axis[None, :]).argmin(axis=1)
    A_idx = np.abs(A_vals[:, None] - A_axis[None, :]).argmin(axis=1)

    for i in range(len(cx_vals)):
        ei, ai = E_idx[i], A_idx[i]
        image[0, ei, ai] = cx_vals[i]
        image[1, ei, ai] = 1.0

    return image

def random_grid():

    E_min, E_max, E_step = utils.random_energy_range()
    A_min, A_max, A_step = utils.random_angle_range()

    E_axis = np.arange(E_min, E_max, E_step)
    A_axis = np.arange(A_min, A_max, A_step)

    return E_axis, A_axis

def crop_image(image):

    global_E_axis, global_A_axis = global_grid()
    random_E_axis, random_A_axis = random_grid()

    for A_idx, A in enumerate(global_A_axis):
        for E_idx, E in enumerate(global_E_axis):
            
            E = round(E, 2)
            A = round(A, 2)

            if not np.any(np.isclose(A, random_A_axis)) or \
                not np.any(np.isclose(E, random_E_axis)):
                image[1, E_idx, A_idx] = 0.0

    return image

def get_images(train_path, log=True):

    # with open(train_path, 'r') as f:
    #     data = json.load(f)

    with gzip.open(train_path, 'rb') as f:
        json_bytes = f.read()
        json_str = json_bytes.decode()
        data = json.loads(json_str)

    n = len(data)
    images = []

    for i in range(n):

        points = data[i]['observable_sets'][0]['points']
        
        E_vals = []
        A_vals = []
        cx_vals = []

        for p in points:

            E_vals.append(p['cn_ex'])
            A_vals.append(p['theta_cm_out'])

            if log == True:
                cx = np.log10(p['dsdO'])
                cx_vals.append(cx)
            else:
                cx_vals.append(p['dsdO'])

        for i in range(IMG_DUP):
            image = place_image_on_grid(E_vals, A_vals, cx_vals)
            image = crop_image(image)
            images.append(image)

    return torch.stack(images, dim=0)

# get all targets from a training set
# returns a tensor of shape [n_samples, 2]
# energy, total width
def get_targets(train_path, dup=True):

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

        if dup:
            for i in range(IMG_DUP):
                tensors.append(torch.tensor([Er_unit, log10_gamma], dtype=torch.float32))
        else:
            tensors.append(torch.tensor([Er_unit, log10_gamma], dtype=torch.float32))

    return torch.stack(tensors, dim=0)

# input dataset for ResonanceCNN
class ResonanceDataset(Dataset):

    def __init__(self, images, targets, gradients=True):

        self.images = images # shape: [N, C=2, E, A]
        self.targets = targets # shape: [N, 2]
        self.gradients = gradients

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        image = self.images[idx]
        target = self.targets[idx]

        if self.gradients:
            image = utils.sobel(image)

        return image, target