import gzip
import json
import torch
import numpy as np
import pandas as pd
import preprocessing
from torch.utils.data import Dataset

MAX_RESONANCES = 20

# returns images, target_params (energies and gammas), target_mask, target_count
def get_images_and_targets(train_path, crop_strength, log_cx=True, compressed=True):

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
    targets = []
    
    for i in range(n):

        print(f'get_images_and_targets starting on {i}...')

        points = data[i]['observable_sets'][0]['points']
        
        # image data
        E_vals = []
        A_vals = []
        cx_vals = []

        # target data
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

        # log total gamma
        log_gamma_total = float(np.log10(gamma_total + 1e-8))

        # add all image data
        for p in points:

            E_vals.append(p['cn_ex'])
            A_vals.append(p['theta_cm_out'])

            if log_cx:
                cx_vals.append(np.log10(p['dsdO']))
            else:
                cx_vals.append(p['dsdO'])

        # duplicate and process (place on grid and crop) each sample
        for j in range(preprocessing.IMG_DUP):
            
            image = preprocessing.place_image_on_grid(E_vals, A_vals, cx_vals)

            if crop_strength > 0.0:

                print(f'cropping {i}, {j}...')

                image, cropped_E_axis, cropped_A_axis = preprocessing.crop_image(image, crop_strength)

                # find which resonances are in the cropped image
                levels = data[i]['levels'][0]
                es = []
                gs = []
                for _ in levels:
                    energy = levels['energy']
                    gamma_total = levels['gamma_total']
                    if cropped_E_axis.min() <= energy <= cropped_E_axis.max():
                        es.append(energy)
                        gs.append(gamma_total)

                order = torch.argsort(es)
                e = es[order]
                g = gs[order]

                # build target tensors
                count = e.numel()
                target_params = torch.zeros(MAX_RESONANCES, 2, dtype=torch.float32)
                target_mask = torch.zeros(MAX_RESONANCES, dtype=torch.bool)
                target_params[:count, 0] = e[:count]
                target_params[:count, 1] = g[:count]
                target_mask[:count] = True

            images.append(image)

    return torch.stack(images, dim=0), \
            target_params, \
            target_mask, \
            torch.tensor(count, dtype=torch.int64)

# input dataset for ResonanceCNN
class ResonanceDataset(Dataset):

    def __init__(self, images, targets, gradients):

        self.images = images
        self.targets = targets
        self.gradients = gradients

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        image = self.images[idx]
        target = self.targets[idx]

        if self.gradients:
            image = preprocessing.sobel(image)

        return image, target