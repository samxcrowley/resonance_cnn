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
        if norm_energies:
            energies = utils.normalise(energies, energy_min, energy_max)

        grid = df.pivot(index='cn_ex', columns='theta_cm_out', values='dsdO') \
                    .reindex(index=sorted(df['cn_ex'].unique()), \
                            columns=sorted(df['theta_cm_out'].unique())).values
        
        EE, TH = torch.meshgrid(
            torch.tensor(energies, dtype=torch.float32),
            torch.tensor(angles, dtype=torch.float32),
            indexing="ij"
        )

        t = torch.stack([
            torch.tensor(grid, dtype=torch.float32), # dsdO (log)
            TH, # angle in [0,1] if norm_angles
            EE, # energy in [0,1] if norm_energies
            torch.zeros_like(EE) # vis. mask
        ], dim=0)  # shape (4, E, A)

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

        # normalise energy
        points = data[i]['observable_sets'][0]['points']
        df = pd.DataFrame(points)
        energies_abs = sorted(df['cn_ex'].unique())
        e_min, e_max = min(energies_abs), max(energies_abs)
        Er_unit = (energy - e_min) / max(e_max - e_min, 1e-8)
        Er_unit = float(np.clip(Er_unit, 0.0, 1.0))

        log10_gamma = float(np.log10(gamma_total + 1e-8))

        tensors.append(torch.tensor([Er_unit, log10_gamma], dtype=torch.float32))

    return torch.stack(tensors, dim=0)

class ResonanceDataset(Dataset):

    def __init__(self, images, targets):

        self.images = images # shape: [n_samples, 4, n_energies, n_angles]
        self.targets = targets # shape: [n_samples, 2]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        image = self.images[idx]
        target = self.targets[idx]

        return image, target