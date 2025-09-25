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
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

import data_loading
import utils

class SingleRes_EnergyLevel_CNN(nn.Module):

    def __init__(self, in_ch=2, base=80, dropout_p=0.3, kernel_size=3):

        super().__init__()

        padmode = 'reflect'

        self.conv1 = nn.Conv2d(in_ch, base, kernel_size, padding='same', padding_mode=padmode)
        self.bn1 = nn.BatchNorm2d(base)

        self.conv2 = nn.Conv2d(base, base * 2, kernel_size, padding='same', padding_mode=padmode)
        self.bn2 = nn.BatchNorm2d(base * 2)

        self.conv3 = nn.Conv2d(base * 2, base * 4, kernel_size, padding='same', padding_mode=padmode)
        self.bn3 = nn.BatchNorm2d(base * 4)

        self.conv4 = nn.Conv2d(base * 4, base * 8, kernel_size, padding='same', padding_mode=padmode)
        self.bn4 = nn.BatchNorm2d(base * 8)

        # TODO: should A be downsampled too? if res. becomes high enough?
        self.pool = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)) # downsample E only

        self.fc1 = nn.Linear(base * 8, 256)
        self.fc2 = nn.Linear(256, 128)

        self.dropout = nn.Dropout(dropout_p)

        self.head_E = nn.Linear(128, 1)

    def forward(self, x):

        # x shape: (N, 2, E, A)
        img = x[:, :1]
        mask = x[:, 1:]

        # stage 1
        h = F.relu(self.bn1(self.conv1(torch.cat([img, mask], dim=1)))) # keep both channels for 1st conv
        h = self.pool(h)
        mask = self.pool(mask) # keep mask in sync (downsample E)

        # stage 2
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.pool(h)
        mask = self.pool(mask)

        # stage 3
        h = F.relu(self.bn3(self.conv3(h)))
        h = self.pool(h)
        mask = self.pool(mask)

        # stage 4
        h = F.relu(self.bn4(self.conv4(h)))
        h = self.pool(h)
        mask = self.pool(mask)

        # masked global average pooling
        # broadcast mask to all channels
        mask_b = (mask > 0).float()

        # sum over spatial, divide by n. valid
        denom = mask_b.sum(dim=(2,3), keepdim=False).clamp_min(1e-6)
        num = (h * mask_b).sum(dim=(2,3), keepdim=False)
        feat = num / denom

        x = F.relu(self.fc1(feat))
        x = self.dropout(F.relu(self.fc2(x)))

        Er_unit = torch.sigmoid(self.head_E(x)).squeeze(-1)

        return Er_unit