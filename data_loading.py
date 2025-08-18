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

        if log:
            df['dsdO'] = np.log10(df['dsdO'])

        grid = df.pivot(index='cn_ex', columns='theta_cm_out', values='dsdO')

        t = torch.tensor(grid.values, dtype=torch.float32)
        t = t.unsqueeze(0)

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
