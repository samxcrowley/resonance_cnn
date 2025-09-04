import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import pandas as pd
import math
import random
import data_loading

def get_axes(image):

    A = torch.unique(image[1, :, :])
    E = torch.unique(image[2, :, :])
    
    return A, E

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
    plt.colorbar(label="cx")
    plt.xlabel("A")
    plt.ylabel("E")
    plt.savefig(f'images/testing/{name}')

def random_crop(image, crop_coef=3, angle_p=0.25):

    E_axis, A_axis = data_loading.global_grid()
    
    num_E = len(E_axis)
    num_A = len(A_axis)

    for A in range(num_A):

        E_bot = random.random() / crop_coef
        E_top = random.random() / crop_coef

        E_min = math.floor(num_E * E_bot)
        E_max = math.ceil(num_E - (num_E * E_top))

        roi = torch.zeros((num_E), dtype=torch.bool, device=image.device)
        roi[E_min:E_max] = True
        outside = ~roi

        if random.random() < angle_p:
            image[1, :, A] = 0
        else:
            image[1, outside, A] = 0

    return image

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
    plt.plot(df["epoch"], df["train_loss_E"], label="train loss (E)")
    plt.plot(df["epoch"], df["train_loss_G"], label="train loss (G)")
    # plt.plot(df["epoch"], df["val_loss"], label="val loss")
    # plt.plot(df["epoch"], df["train_mae_E"], label="MAE")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Total loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/{name}', dpi=150)