import json
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset

def get_images(train_path, log=True):

    with open(train_path, 'r') as f:
        data = json.load(f)

    n = len(data)
    tensors = []
    for i in range(n):

        points = data[i]['observable_sets'][0]['points']

        df = pd.DataFrame(points)
        df = df.drop(columns=['ke_cm_in', 'dsdRuth', 'dsdO-dsRuth'])

        angles = sorted(df['theta_cm_out'].unique())
        energies = sorted(df['cn_ex'].unique())

        grid = df.pivot(index='cn_ex', columns='theta_cm_out', values='dsdO')\
            .reindex(index=energies, columns=angles).values
        
        xx, yy = torch.meshgrid(torch.tensor(angles), torch.tensor(energies), indexing="xy")

        if log:
            df['dsdO'] = np.log10(df['dsdO'])

        # energies = grid.index.values
        # angles = grid.columns.values
        # E, A = np.meshgrid(energies, angles, indexing='ij')

        # cross_section = grid.values
        # energy_channel = np.tile(E[..., None], (1,1))  # shape (E, A)
        # angle_channel = np.tile(A[None, ...], (len(energies),1))

        # print(energy_channel)
        # print(angle_channel)

        # # Stack into channels
        # t = np.stack([cross_section, energy_channel, angle_channel], axis=0)
        # t = torch.tensor(t, dtype=torch.float32)  # shape (3, E, A)

        # print(t.shape)
        # break

        # t = torch.tensor(grid.values, dtype=torch.float32)
        # t = t.unsqueeze(0)

        t = torch.stack([
            torch.tensor(grid, dtype=torch.float32),
            xx.float(),
            yy.float()
        ], dim=0)

        tensors.append(t)

    return torch.stack(tensors, dim=0)

class ResonanceDataset(Dataset):

    def __init__(self, images, targets):

        # store cross-sectional value as a function of (E, theta)
        self.images = images # shape: (num_samples, 1, num_theta, num_E)

        # targets stores a set of resonance data for each sample
        # consists of: energy, width (Gamma), dominant l
        self.targets = targets # shape: (num_samples, 3)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        image = self.images[idx]
        target = {key: val[idx] for key, val in self.targets.items()}

        return image, target
