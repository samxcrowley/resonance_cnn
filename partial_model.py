import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class PartialConv2D(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding='same', bias=True):

        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias

        if padding == 'same':
            self.padding = self.kernel_size // 2

        # scaling factor -- number of params. in a window
        self.K = in_ch * kernel_size * kernel_size

        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        # nn.init.ones_(self.weight)

        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, mask):

        N = x.size(0)
        C = x.size(1)
        E = x.size(2)
        A = x.size(3)

        # broadcast mask along channels if needed
        if mask.size(1) == 1 and C > 1:
            mask = mask.repeat(1, C, 1, 1)

        p = self.padding
        x_pad = F.pad(x, (p, p, p, p), mode='constant', value=0.0)
        mask_pad = F.pad(mask, (p, p, p, p), mode='constant', value=1.0)

        x_unfold = F.unfold(x_pad, \
                            kernel_size=(self.kernel_size, self.kernel_size), \
                            stride=(self.stride, self.stride))
        mask_unfold = F.unfold(mask_pad, \
                            kernel_size=(self.kernel_size, self.kernel_size), \
                            stride=(self.stride, self.stride))
        
        valid_count = mask_unfold.sum(dim=1, keepdim=True)

        x_mask_unfold = x_unfold * mask_unfold

        weight_flat = self.weight.view(self.out_ch, -1)

        # convolution
        y_flat = torch.einsum('ok,nkl->nol', weight_flat, x_mask_unfold)

        # scale
        scale = (self.K / valid_count.clamp(min=1.0))
        y_flat = y_flat * scale

        if self.bias is not None:
            y_flat = y_flat + self.bias.view(1, -1, 1)

        no_valid = (valid_count <= 0)
        if no_valid.any():
            y_flat = y_flat.masked_fill(no_valid.expand_as(y_flat), 0.0)

        # reshape back to (N, Cout, E, A)
        E_out = (E + 2 * p - self.kernel_size) // self.stride + 1
        A_out = (A + 2 * p - self.kernel_size) // self.stride + 1
        y = y_flat.view(N, self.out_ch, E_out, A_out)

        mask_next = (~no_valid).to(x.dtype).view(N, 1, E_out, A_out)

        return y, mask_next

class ResonancePartialCNN(nn.Module):

    def __init__(self, in_ch=1, base=80, dropout_p=0.3, kernel_size=3):

        super().__init__()

        self.p1 = PartialConv2D(in_ch=in_ch, out_ch=base, kernel_size=kernel_size)
        # self.gn1 = nn.GroupNorm(8, base)
        self.gn1 = nn.BatchNorm2d(base)

        self.p2 = PartialConv2D(in_ch=base, out_ch=base*2, kernel_size=kernel_size)
        # self.gn2 = nn.GroupNorm(8, base*2)
        self.gn2 = nn.BatchNorm2d(base*2)

        self.p3 = PartialConv2D(in_ch=base*2, out_ch=base*4, kernel_size=kernel_size)
        # self.gn3 = nn.GroupNorm(8, base*4)
        self.gn3 = nn.BatchNorm2d(base*4)

        self.p4 = PartialConv2D(in_ch=base*4, out_ch=base*8, kernel_size=kernel_size)
        # self.gn4 = nn.GroupNorm(8, base*8)
        self.gn4 = nn.BatchNorm2d(base*8)

        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1)) # size-invariant

        self.fc1 = nn.Linear(base*8, 256)
        self.fc2 = nn.Linear(256, 128)

        self.dropout = nn.Dropout(dropout_p)
        
        self.head_E = nn.Linear(128, 1)
        # self.head_G = nn.Linear(128, 1)

    def forward(self, x):

        data = x[:, 0:1, :, :]
        mask = x[:, 1:2, :, :].clamp(0, 1)
        x = data

        # block 1
        x, mask = self.p1(x, mask)
        x = F.relu(self.gn1(x))
        x = self.pool(x)
        mask = self.pool(mask)

        # block 2
        x, mask = self.p2(x, mask)
        x = F.relu(self.gn2(x))
        x = self.pool(x)
        mask = self.pool(mask)

        # block 3
        x, mask = self.p3(x, mask)
        x = F.relu(self.gn3(x))
        x = self.pool(x)
        mask = self.pool(mask)

        # block 4
        x, mask = self.p4(x, mask)
        x = F.relu(self.gn4(x))
        x = self.pool(x)
        mask = self.pool(mask)

        feat = self.gap(x).flatten(1)

        z = F.relu(self.fc1(feat))
        z = self.dropout(F.relu(self.fc2(z)))

        Er_unit = torch.sigmoid(self.head_E(z)).squeeze(-1)
        # logGamma = self.head_G(z).squeeze(-1)

        # return Er_unit, logGamma
        return Er_unit