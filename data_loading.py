import json
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset

import utils

# get all images from a training set
# returns a tensor of shape [n_samples, 4, n_energies, n_angles]
# 4 channels in dim. 1 are: cross-section, angle, energy, vis. mask
def get_images(train_path, log=True, norm_angles=True, norm_energies=True):

    with open(train_path, 'r') as f:
        data = json.load(f)

    n = len(data)
    tensors = []

    for i in range(n):

        points = data[i]['observable_sets'][0]['points']
        df = pd.DataFrame(points)
        df = df.drop(columns=['ke_cm_in', 'dsdRuth', 'dsdO-dsRuth'])

        if log:
            df['dsdO'] = np.log10(df['dsdO'])

        angles = sorted(df['theta_cm_out'].unique())
        angle_min = min(angles)
        angle_max = max(angles)
        if norm_angles:
            angles = utils.normalise(angles, angle_min, angle_max)

        energies = sorted(df['cn_ex'].unique())
        energy_min = min(energies)
        energy_max = max(energies)
        if norm_angles:
            energies = utils.normalise(energies, energy_min, energy_max)

        grid = df.pivot(index='cn_ex', columns='theta_cm_out', values='dsdO') \
                    .reindex(index=sorted(df['cn_ex'].unique()), \
                            columns=sorted(df['theta_cm_out'].unique())).values
        
        xx, yy = torch.meshgrid(torch.tensor(angles), torch.tensor(energies), indexing="xy")

        t = torch.stack([
            torch.tensor(grid, dtype=torch.float32),
            xx.float(),
            yy.float(),
            torch.zeros_like(xx)
        ], dim=0)

        tensors.append(t)

    return torch.stack(tensors, dim=0)

def get_targets(train_path):

    with open(train_path, 'r') as f:
        data = json.load(f)

    n = len(data)
    tensors = []
    for i in range(n):

        levels = data[i]['levels'][0]
        energy = levels['energy']
        gamma_total = levels['Gamma_total']

        tensors.append(torch.tensor([energy, gamma_total]))

    return torch.stack(tensors, dim=0)

class ResonanceDataset(Dataset):

    def __init__(self, images, targets):

        self.images = images # shape: [n_samples, 4, n_energies, n_angles]
        self.targets = targets # shape: (n_samples, 2)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        image = self.images[idx]
        target = {key: val[idx] for key, val in self.targets.items()}

        return image, target