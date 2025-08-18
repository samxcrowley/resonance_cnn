import numpy as np
import matplotlib.pyplot as plt
import torch

def get_axes(images):

    subset = images[:, 1:3, :, :]

    angles = images[:, 1, :, :].reshape(-1)
    energies = images[:, 2, :, :].reshape(-1)
    unique_angles = torch.unique(angles)
    unique_energies = torch.unique(energies)
    
    return unique_angles, unique_energies

def plot_image(images, idx):

    image = images[idx]
    xs = image[0].numpy()

    unique_angles, unique_energies = get_axes(images)

    plt.pcolormesh(unique_angles, unique_energies, xs)

    plt.show()