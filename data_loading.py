import json
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

def build_e_theta_axes(points):

    angles = set()
    energies = set()
    
    for pt in points:
        angles.add(pt['theta_cm_out'])
        energies.add(pt['ke_cm_in'])

    angles = np.array(sorted(angles), dtype=np.float32)
    energies = np.array(sorted(energies), dtype=np.float32)
    
    return angles, energies

def build_sample():
    pass

def points_to_tensor(samples, angles, energies):

    N = len(samples)
    A = len(angles)
    E = len(energies)

    ang_to_idx = {val: i for i, val in enumerate(angles)}
    en_to_idx  = {val: i for i, val in enumerate(energies)}
    
    X = np.full((N, 2, A, E), 0.0, dtype=np.float32)
    
    # set mask channel initially 0
    X[:, 1, :, :] = 0.0

    # fill from points
    for n, pts in enumerate(samples):
        for p in pts:
            a = ang_to_idx.get(p["theta_cm_out"])
            e = en_to_idx.get(p["ke_cm_in"])
            if a is None or e is None:
                # point not on the global axes; skip or handle tolerance matching
                continue
            X[n, 0, a, e] = np.float32(p["dsdo"])
            X[n, 1, a, e] = 1.0

        # If you truly have perfect coverage for every coordinate, you can also do:
        # X[n, 1, :, :] = 1.0

    return torch.from_numpy(X)

def load_images(train_path):
    
    with open(train_path, 'r') as f:
        data = json.load(f)
    
    n = len(data)
    
    for i in range(n):

        levels = data[i]['levels'][0]
        observable_sets = data[i]['observable_sets'][0]
    
        if i >= 1:
            continue

        # currently only works for a resonance containing one in/out pairing
        pp_in_index = observable_sets['pp_in_index']
        pp_out_index = observable_sets['pp_out_index']
        points = observable_sets['points']

        angles, energies = build_e_theta_axes(points)

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
