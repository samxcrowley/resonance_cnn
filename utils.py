import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import pandas as pd
import os
import data_loading

def random_range(minimum, maximum, step, strength):

    axis = np.linspace(minimum, \
                         maximum, \
                         int((maximum - minimum) / step) + 1)
    
    n = len(axis)

    subset_length_minimum = n * (0.5 * (1 - strength))
    subset_length_maximum = n * (1 - strength)

    subset_length = int(np.random.uniform(subset_length_minimum, subset_length_maximum))

    minimum_upper_limit = maximum - (subset_length * step)
    
    minimum_ = round(np.random.uniform(minimum, minimum_upper_limit), 1)
    maximum_ = round(minimum_ + (subset_length * step), 1)

    return minimum_, maximum_

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

    filename = f'plots/images/{name}.png'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=150)

def plot_results(results_name):

    df = pd.read_csv(f'results/{results_name}.csv')

    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="train loss")
    plt.plot(df["epoch"], df["val_loss"], label="val loss")
    plt.plot(df["epoch"], df["train_mae_E"], label="MAE")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Total loss")
    plt.legend()
    plt.tight_layout()

    filename = f'plots/results/{results_name}.png'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=150)

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