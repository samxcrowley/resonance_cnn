import torch
import torch.nn as nn
import torch.nn.functional as F

class ResonanceCNN(nn.Module):

    def __init__(self, in_ch=2, base=80, dropout_p=0.3, kernel_size=3):

        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, base, kernel_size=kernel_size, padding='same')
        # self.bn1 = nn.GroupNorm(8, base)
        self.bn1 = nn.BatchNorm2d(base)

        self.conv2 = nn.Conv2d(base, base * 2, kernel_size=kernel_size, padding='same')
        # self.bn2 = nn.GroupNorm(8, base * 2)
        self.bn2 = nn.BatchNorm2d(base * 2)

        self.conv3 = nn.Conv2d(base * 2, base * 4, kernel_size=kernel_size, padding='same')
        # self.bn3 = nn.GroupNorm(8, base * 4)
        self.bn3 = nn.BatchNorm2d(base * 4)

        self.conv4 = nn.Conv2d(base * 4, base * 8, kernel_size=kernel_size, padding='same')
        # self.bn4 = nn.GroupNorm(8, base * 8)
        self.bn4 = nn.BatchNorm2d(base * 8)

        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)) # downsample E only
        self.gap = nn.AdaptiveAvgPool2d((1, 1)) # size-invariant

        self.fc1 = nn.Linear(base * 8, 256)
        self.fc2 = nn.Linear(256, 128)

        self.dropout = nn.Dropout(dropout_p)

        self.head_E = nn.Linear(128, 1) # Er_unit -> sigmoid in forward
        # self.head_G = nn.Linear(128, 1) # logGamma -> linear

    def forward(self, x):

        # x shape: (N, 2, E, A)
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) # (N, base, E/2, A)
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) # (N, 2 * base, E/4, A)
        x = self.pool(F.relu(self.bn3(self.conv3(x)))) # (N, 4 * base, E/8, A)
        x = self.pool(F.relu(self.bn4(self.conv4(x)))) # (N, 8 * base, E/16, A)
        x = self.gap(x).flatten(1) # (N, 8 * base)

        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))

        Er_unit = torch.sigmoid(self.head_E(x)).squeeze(-1) # (N,)
        # logGamma = self.head_G(x).squeeze(-1) # (N,)

        # return Er_unit, logGamma
        return Er_unit