import numpy as np
import matplotlib.pyplot as plt
import torch

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