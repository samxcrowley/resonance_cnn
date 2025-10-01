import gzip
import json
import torch
import numpy as np
import pandas as pd
import preprocessing
from torch.utils.data import Dataset

def get_images(train_path, crop_strength, log=True, compressed=True):

    if compressed:
        with gzip.open(train_path, 'rb') as f:
            json_bytes = f.read()
            json_str = json_bytes.decode()
            data = json.loads(json_str)
    else:
        with open(train_path, 'r') as f:
            data = json.load(f)

    n = len(data)
    images = []
    
    for i in range(n):

        print(f'get_images starting on {i}...')

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

        for j in range(preprocessing.IMG_DUP):
            
            image = preprocessing.place_image_on_grid(E_vals, A_vals, cx_vals)

            if crop_strength > 0.0:
                print(f'cropping {i}, {j}...')
                image = preprocessing.crop_image(image, crop_strength)

            images.append(image)

    return torch.stack(images, dim=0)

# get all targets from a training set
# returns a tensor of shape [n_samples, 2]
# energy, total width
def get_targets(train_path, compressed=True):

    if compressed:
        with gzip.open(train_path, 'rb') as f:
            json_bytes = f.read()
            json_str = json_bytes.decode()
            data = json.loads(json_str)
    else:
        with open(train_path, 'r') as f:
            data = json.load(f)

    n = len(data)
    tensors = []
    for i in range(n):

        levels = data[i]['levels'][0]
        energy = levels['energy']
        gamma_total = levels['Gamma_total']
        num_levels = len(data[i]['levels'])

        # normalise energy
        points = data[i]['observable_sets'][0]['points']
        df = pd.DataFrame(points)
        energies_abs = sorted(df['cn_ex'].unique())
        e_min, e_max = min(energies_abs), max(energies_abs)
        Er_unit = (energy - e_min) / max(e_max - e_min, 1e-8)
        Er_unit = float(np.clip(Er_unit, 0.0, 1.0))

        log10_gamma = float(np.log10(gamma_total + 1e-8))

        # duplicate targets to match duplicated (cropped) images
        for i in range(preprocessing.IMG_DUP):
            tensors.append(torch.tensor([Er_unit, log10_gamma, num_levels], dtype=torch.float32))

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
            image = preprocessing.sobel(image)

        return image, target