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

    images = data_loading.get_images(path, log=True) # shape: [1000, 1, 101, 7]

    idx = 0
    image = images[idx]
    xs = image[0].numpy()
    angle = np.arange(xs.shape[1] + 1)
    energy = np.arange(xs.shape[0] + 1)
    plt.pcolormesh(angle, energy, xs)
    plt.show()

    # heatmap = df.pivot(index='cn_ex', columns='theta_cm_out', values='dsdO')

    # plt.figure(figsize=(10, 6))

    # plt.imshow(
    #     heatmap.values,
    #     origin='lower',
    #     aspect='auto',
    #     cmap='viridis',
    #     extent=[
    #         heatmap.columns.min(), heatmap.columns.max(),
    #         heatmap.index.min(), heatmap.index.max()
    #     ]
    # )

    # plt.colorbar(label='Cross Section (dsdO)')
    # plt.xlabel('Scattering Angle (degrees)')
    # plt.ylabel('Excitation Energy (MeV)')
    # plt.title('Differential Cross Section Heatmap')
    # plt.show()