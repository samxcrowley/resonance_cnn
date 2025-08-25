import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def get_axes(images):

    angles = images[:, 1, :, :].reshape(-1)
    energies = images[:, 2, :, :].reshape(-1)
    unique_angles = torch.unique(angles)
    unique_energies = torch.unique(energies)
    
    return unique_angles, unique_energies

# normalise a tensor x to [0, 1]
def normalise(x, min_val=None, max_val=None):

    if min_val is None:
        min_val = x.min()
    if max_val is None:
        max_val = x.max()
    return (x - min_val) / (max_val - min_val + 1e-8)

def plot_image(images, idx):

    image = images[idx]
    xs = image[0].numpy()

    unique_angles, unique_energies = get_axes(images)

    plt.pcolormesh(unique_angles, unique_energies, xs)

    plt.show()

def plot_single_image(image):
    plot_image(image.unsqueeze(0), 0)

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