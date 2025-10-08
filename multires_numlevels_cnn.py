import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiRes_NumLevels_CNN(nn.Module):

    def __init__(self, in_ch=2, base=80, dropout_p=0.3, kernel_size=3, max_levels=5):

        super().__init__()

        padmode = 'reflect'

        self.conv1 = nn.Conv2d(in_ch, base, kernel_size, padding='same', padding_mode=padmode)
        self.bn1 = nn.GroupNorm(8, base)

        self.conv2 = nn.Conv2d(base, base * 2, kernel_size, padding='same', padding_mode=padmode)
        self.bn2 = nn.GroupNorm(8, base * 2)

        self.conv3 = nn.Conv2d(base * 2, base * 4, kernel_size, padding='same', padding_mode=padmode)
        self.bn3 = nn.GroupNorm(8, base * 4)

        self.conv4 = nn.Conv2d(base * 4, base * 8, kernel_size, padding='same', padding_mode=padmode)
        self.bn4 = nn.GroupNorm(8, base * 8)

        # TODO: should A be downsampled too? if res. becomes high enough?
        self.pool = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)) # downsample E only

        self.fc1 = nn.Linear(base * 8, 256)
        self.fc2 = nn.Linear(256, 128)

        self.dropout = nn.Dropout(dropout_p)

        # num_levels head
        self.head_num_levels = nn.Linear(128, 1)

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
        
        pred = self.head_num_levels(x).squeeze(-1)

        return pred

class MultiRes_NumLevels_SmallCNN(nn.Module):

    def __init__(self, in_ch=2, base=32, dropout_p=0.1, kernel_size=3, max_levels=5):

        super().__init__()
        padmode = 'reflect'

        self.conv1 = nn.Conv2d(in_ch, base, kernel_size, padding='same', padding_mode=padmode)
        self.bn1 = nn.GroupNorm(8, base)

        self.conv2 = nn.Conv2d(base, base * 2, kernel_size, padding='same', padding_mode=padmode)
        self.bn2 = nn.GroupNorm(8, base * 2)

        self.pool = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))

        self.dropout = nn.Dropout(dropout_p)
        self.head = nn.Linear(base*2, max_levels)

    def forward(self, x):

        # x: (N, 2, E, A)
        img  = x[:, :1]
        mask = x[:, 1:]  # (N,1,E,A)

        h = F.relu(self.bn1(self.conv1(torch.cat([img, mask], dim=1))))
        h = self.pool(h)

        h = F.relu(self.bn2(self.conv2(h)))
        h = self.pool(h)

        # --- masked GAP ---
        # Resize mask once to match h's spatial size
        mask_ds = F.interpolate(mask, size=h.shape[-2:], mode='nearest')
        mask_b  = (mask_ds > 0).float()

        denom = mask_b.sum(dim=(2,3)).clamp_min(1e-6)   # (N,1)
        num   = (h * mask_b).sum(dim=(2,3))             # (N,C)
        feat  = num / denom                              # broadcast → (N,C)

        feat  = self.dropout(feat)
        logits = self.head(feat)                         # (N, max_levels)
        return logits