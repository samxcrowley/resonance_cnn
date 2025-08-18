import gzip
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

import data_loading

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    path = 'data/o16/o16_training.gz'

    images = data_loading.get_images(path, log=True) # shape: [1000, 3, 101, 7]
    
    # get min and max for energy and angle
    subset = images[:, 1:3, :, :]
    min_vals = subset.amin(dim=(0,2,3,)).squeeze()
    max_vals = subset.amax(dim=(0,2,3,)).squeeze()
    angle_min = min_vals[0].item()
    angle_max = max_vals[0].item()
    energy_min = min_vals[1].item()
    energy_max = max_vals[1].item()

    idx = 0
    image = images[idx]
    xs = image[0].numpy()
    angle = np.arange(xs.shape[1] + 1)
    energy = np.arange(xs.shape[0] + 1)
    plt.pcolormesh(angle, energy, xs)
    # plt.xlim(angle_min, angle_max)
    # plt.ylim(energy_min, energy_max)
    plt.show()