import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import pandas as pd
import math
import random
import data_loading

def pick_by_dist(vals, probs):

    n = len(probs)
    split_size = int(len(vals) / n)

    p = np.random.random()

    for i in range(len(probs)):
        _p = probs[i]
        if p < _p:
            split_start = i * split_size
            split_end = split_start + split_size
            return random.choice(vals[split_start:split_end])

def random_energy_range():

    E_step = random.choice([0.1, 0.2, 0.3, 0.4, 0.5])
    vals = np.arange(data_loading.global_E_min,
                     data_loading.global_E_max,
                     data_loading.global_E_step)
    n = len(vals)

    E_min = round(pick_by_dist(vals[:int(n / 2)], [0.75, 1.0]), 2)
    E_max = round(pick_by_dist(vals[int(n / 2):], [0.75, 0.9, 1.0]), 2)

    return E_min, E_max, E_step

def random_angle_range():

    n = random.choice([3, 4, 5, 6, 7])

    angles = np.arange(data_loading.global_A_min,
                       data_loading.global_A_max,
                       data_loading.global_A_step)
    
    A_max = random.choice(angles[int(len(angles) / 2):])
    A_min = A_max - (data_loading.global_A_step * n)

    return A_min, A_max, data_loading.global_A_step

def plot_image(image, name):

    E_axis, A_axis = data_loading.global_grid()

    values = image[0].numpy() if isinstance(image, torch.Tensor) else image[0]
    mask = image[1].numpy() if isinstance(image, torch.Tensor) else image[1]

    # mask out invalid values
    plot_data = np.where(mask == 1, values, np.nan)

    plt.figure(figsize=(6, 4))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='black')

    plt.pcolormesh(A_axis, E_axis, plot_data, cmap=cmap, shading='auto')
    plt.colorbar(label="dsdO")
    plt.xlabel("Angle")
    plt.ylabel("Energy")
    plt.savefig(f'images/{name}')

def sobel(image):
    
    device = image.device
    dtype = image.dtype

    x = image[0].unsqueeze(0).unsqueeze(0)

    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=dtype, device=device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=dtype, device=device).view(1, 1, 3, 3)

    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)

    grad = torch.sqrt(gx * gx + gy * gy + 1e-12)
    out = image.clone()
    out[0] = grad[0]

    return out

# normalise a tensor x to [0, 1]
def normalise(x, min_val=None, max_val=None):

    if min_val is None:
        min_val = x.min()
    if max_val is None:
        max_val = x.max()
    return (x - min_val) / (max_val - min_val + 1e-8)

def plot_training_data(train_data, name):

    df = pd.read_csv(train_data)

    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="train loss")
    plt.plot(df["epoch"], df["val_loss"], label="val loss")
    plt.plot(df["epoch"], df["train_mae_E"], label="MAE")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Total loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/{name}', dpi=150)