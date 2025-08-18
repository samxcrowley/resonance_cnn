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
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

import data_loading

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

data_loading.load_images('data/o16/o16_training.gz')
