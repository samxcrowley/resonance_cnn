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
import utils

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    path = 'data/o16/o16_training.gz'

    # images = data_loading.get_images(path, log=True) # shape: [1000, 4, 101, 7]
    # utils.plot_image(images, 2)

    targets = data_loading.get_targets(path)
    print(targets.shape)
    print(targets[0])