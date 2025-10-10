import gzip
import json
import math
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

        points = data['data'][i]['observable_sets'][0]['points']
        
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
            for lev in range(len(levels)):

                level = levels[lev]

                energy = level['energy']
                gamma_total = level['Gamma_total']

                # only keep resonances within the cropped range
                if cropped_E_axis.min() <= energy <= cropped_E_axis.max():

                    # normalise energy
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

def save_images_and_targets(nr, strength):

    train_path = f'data/o16/{nr}res_training.gz'

    images, target_params, target_masks = \
        get_images_and_targets(train_path, crop_strength=strength)

    torch.save(images, f'data/images/{nr}res_images_crop{strength}.pt')
    torch.save(target_params, f'data/targets/{nr}res_targetparams_crop{strength}.pt')
    torch.save(target_masks, f'data/targets/{nr}res_targetmasks_crop{strength}.pt')

def load_images_and_targets(nr, strength):

    images = torch.load(f'data/images/{nr}res_images_crop{strength}.pt')
    target_params = torch.load(f'data/targets/{nr}res_targetparams_crop{strength}.pt')
    target_masks = torch.load(f'data/targets/{nr}res_targetmasks_crop{strength}.pt')

    return images, target_params, target_masks

# load one experimental image
def get_exp_image(path, compressed=True, log_cx=True):

    if compressed:
        with gzip.open(path, 'rb') as f:
            json_bytes = f.read()
            json_str = json_bytes.decode()
            data = json.loads(json_str)
    else:
        with open(path, 'r') as f:
            data = json.load(f)

    points = data['data'][0]['points']

    # image data
    E_vals = []
    A_vals = []
    cx_vals = []

    # add all image data
    for p in points:

        E_vals.append(p['cn_ex'])
        A_vals.append(p['theta_cm_out'])

        ds = p['dsdO']

        if log_cx:
            cx_vals.append(np.log10(ds))
        else:
            cx_vals.append(ds)

    print(min(E_vals))
    print(max(E_vals))

    image = preprocessing.place_image_on_grid(E_vals, A_vals, cx_vals)

    return image

def get_subset_of_image(img, centre_energy, E_vals, width):

    E_vals = torch.tensor(E_vals)

    E_min = centre_energy - (width / 2)
    E_max = centre_energy + (width / 2)

    inside_E = (E_vals >= E_min) & (E_vals <= E_max) # shape [E]
    inside_2d = inside_E.view(-1, 1).expand(-1, img.shape[2]) # [E, A]

    img_windowed = img.clone()
    img_windowed[1] = img_windowed[1] * inside_2d 
    
    return img_windowed

# input dataset for ResonanceCNN
class ResonanceDataset(Dataset):

    def __init__(self, images, target_params, target_masks, gradients):

        self.images = images
        self.target_params = target_params
        self.target_masks = target_masks
        self.gradients = gradients

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        image = self.images[idx]
        target_params = self.target_params[idx]
        target_mask = self.target_masks[idx]

        if self.gradients:
            image = preprocessing.sobel(image)

        return image, target_params, target_mask