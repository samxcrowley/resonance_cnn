import gzip
import json
import torch
import numpy as np
import pandas as pd
import preprocessing
from torch.utils.data import Dataset

MAX_RESONANCES = 5

# returns images, target_params (energies and gammas), target_mask, target_count
def get_images_and_targets(train_path, crop_strength, log_cx=True, compressed=True, subset=-1):

    if compressed:
        with gzip.open(train_path, 'rb') as f:
            json_bytes = f.read()
            json_str = json_bytes.decode()
            data = json.loads(json_str)
    else:
        with open(train_path, 'r') as f:
            data = json.load(f)

    n = len(data)
    if subset > 0:
        n = subset

    images = []
    target_params = []
    target_masks = []

    E_axis, A_axis = preprocessing.global_grid()
    
    for i in range(n):

        print(f'get_images_and_targets starting on {i}...')

        points = data[i]['observable_sets'][0]['points']
        
        # image data
        E_vals = []
        A_vals = []
        cx_vals = []

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

            if crop_strength == 0.0:
                cropped_E_axis = E_axis
                cropped_A_axis = A_axis
            else:
                print(f'cropping {i}, {j}...')
                image, cropped_E_axis, cropped_A_axis = \
                        preprocessing.crop_image(image, crop_strength)
            
            levels = data[i]['levels']

            es = []
            gs = []
            for i in range(len(levels)):

                level = levels[i]

                energy = level['energy']
                gamma_total = level['Gamma_total']

                # only keep resonances within the cropped range
                if cropped_E_axis.min() <= energy <= cropped_E_axis.max():

                    # normalise energy
                    points = data[i]['observable_sets'][0]['points']
                    Er_unit = (energy - E_axis.min()) / max(E_axis.max() - E_axis.min(), 1e-8)
                    Er_unit = float(np.clip(Er_unit, 0.0, 1.0))
                    es.append(Er_unit)

                    # append log gamma_total
                    gs.append(float(np.log10(gamma_total + 1e-8)))

            # order by energy
            es = torch.tensor(es)
            gs = torch.tensor(gs)
            order = torch.argsort(es)
            e = es[order]
            g = gs[order]

            # build target tensors
            t_count = e.numel()
            t_params = torch.zeros(MAX_RESONANCES, 2, dtype=torch.float32)
            t_mask = torch.zeros(MAX_RESONANCES, dtype=torch.bool)
            t_params[:t_count, 0] = e[:t_count]
            t_params[:t_count, 1] = g[:t_count]
            t_mask[:t_count] = True

            images.append(image)
            target_params.append(t_params)
            target_masks.append(t_mask)

    return torch.stack(images, dim=0), \
            torch.stack(target_params, dim=0), \
            torch.stack(target_masks, dim=0)

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