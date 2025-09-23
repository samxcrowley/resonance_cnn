import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# class PartialConv2D(nn.Module):

#     def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding='same', bias=True):

#         super().__init__()

#         self.in_ch = in_ch
#         self.out_ch = out_ch
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.bias = bias

#         if padding == 'same':
#             self.padding = self.kernel_size // 2

#         # scaling factor -- number of params. in a window
#         self.K = in_ch * kernel_size * kernel_size

#         self.weight = nn.Parameter(torch.empty(out_ch, in_ch, kernel_size, kernel_size))
#         nn.init.kaiming_normal_(self.weight, nonlinearity='relu')

#         self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None
#         if self.bias is not None:
#             nn.init.zeros_(self.bias)

#     def forward(self, x, mask):

#         N = x.size(0)
#         C = x.size(1)
#         E = x.size(2)
#         A = x.size(3)

#         # broadcast mask along channels if needed
#         if mask.size(1) == 1 and C > 1:
#             mask = mask.repeat(1, C, 1, 1)

#         p = self.padding
#         x_pad = F.pad(x, (p, p, p, p), mode='constant', value=0.0)
#         mask_pad = F.pad(mask, (p, p, p, p), mode='constant', value=1.0)

#         x_unfold = F.unfold(x_pad, \
#                             kernel_size=(self.kernel_size, self.kernel_size), \
#                             stride=(self.stride, self.stride))
#         mask_unfold = F.unfold(mask_pad, \
#                             kernel_size=(self.kernel_size, self.kernel_size), \
#                             stride=(self.stride, self.stride))
        
#         valid_count = mask_unfold.sum(dim=1, keepdim=True)

#         x_mask_unfold = x_unfold * mask_unfold

#         weight_flat = self.weight.view(self.out_ch, -1)

#         # convolution
#         y_flat = torch.einsum('ok,nkl->nol', weight_flat, x_mask_unfold)

#         # scale
#         scale = (self.K / valid_count.clamp(min=1.0))
#         y_flat = y_flat * scale

#         if self.bias is not None:
#             y_flat = y_flat + self.bias.view(1, -1, 1)

#         no_valid = (valid_count <= 0)
#         if no_valid.any():
#             y_flat = y_flat.masked_fill(no_valid.expand_as(y_flat), 0.0)

#         # reshape back to (N, Cout, E, A)
#         E_out = (E + 2 * p - self.kernel_size) // self.stride + 1
#         A_out = (A + 2 * p - self.kernel_size) // self.stride + 1
#         y = y_flat.view(N, self.out_ch, E_out, A_out)

#         mask_next = (~no_valid).to(x.dtype).view(N, 1, E_out, A_out)

#         return y, mask_next

class PartialConv2D(nn.Module):
    """
    Robust partial conv that collapses to Conv2d when mask==ones.
    - treat_padding_as_hole=False  -> padding acts like valid zeros (matches Conv2d semantics)
    - use_coverage=True            -> scales by local kernel coverage so edges are not over-amplified
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1,
                 padding='same', bias=True,
                 treat_padding_as_hole=False, use_coverage=True):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_coverage = use_coverage
        self.treat_padding_as_hole = treat_padding_as_hole

        if padding == 'same':
            self.padding = kernel_size // 2
        else:
            self.padding = int(padding)

        # Max window size across channels
        self.K = in_ch * kernel_size * kernel_size

        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None

        # ones kernel (1, C, k, k) – sums valids across channels and kernel window
        self.register_buffer('ones_kernel', torch.ones(1, in_ch, kernel_size, kernel_size))

        # what value to pad the mask with at image borders
        self.pad_mask_value = 0.0 if treat_padding_as_hole else 1.0

    def forward(self, x, mask):
        """
        x:    (N, C, H, W)
        mask: (N, 1 or C, H, W), binary {0,1} ideally
        returns: y, mask_next
        """
        N, C, H, W = x.shape
        assert C == self.in_ch, f"Expected {self.in_ch} input channels, got {C}"

        # broadcast mask to C channels if needed
        if mask.size(1) == 1 and C > 1:
            mask = mask.repeat(1, C, 1, 1)
        elif mask.size(1) != C:
            raise ValueError(f"Mask must have 1 or {C} channels, got {mask.size(1)}")

        p = self.padding
        if p > 0:
            x_pad    = F.pad(x,    (p, p, p, p), value=0.0)
            mask_pad = F.pad(mask, (p, p, p, p), value=self.pad_mask_value)
        else:
            x_pad, mask_pad = x, mask

        # valid_count: how many valid entries in each window (sum across C and kxk)
        valid_count = F.conv2d(mask_pad, self.ones_kernel, bias=None,
                               stride=self.stride, padding=0)  # (N,1,Hout,Wout)

        # masked conv (no bias yet)
        y = F.conv2d(x_pad * mask_pad, self.weight, bias=None,
                     stride=self.stride, padding=0)           # (N,O,Hout,Wout)

        # scale: either by local coverage/valids (edge aware) or K/valids (classic)
        if self.use_coverage:
            # local kernel coverage (how many kernel positions overlap the image)
            ones = torch.ones_like(mask)                      # (N,C,H,W)
            if p > 0:
                ones = F.pad(ones, (p, p, p, p), value=0.0)   # outside image = 0
            coverage = F.conv2d(ones, self.ones_kernel, bias=None,
                                stride=self.stride, padding=0)  # (N,1,Hout,Wout)
            scale = coverage / valid_count.clamp(min=1.0)
        else:
            # classic partial conv scaling
            scale = (self.K / valid_count.clamp(min=1.0))

        y = y * scale

        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)

        # any window with zero valid pixels -> zero output and 0 mask
        no_valid = (valid_count <= 0)
        if no_valid.any():
            y = y.masked_fill(no_valid.expand_as(y), 0.0)

        Hout = (H + 2 * p - self.kernel_size) // self.stride + 1
        Wout = (W + 2 * p - self.kernel_size) // self.stride + 1
        mask_next = (~no_valid).to(x.dtype).view(N, 1, Hout, Wout)

        return y, mask_next

class ResonancePartialCNN(nn.Module):

    def __init__(self, in_ch=1, base=80, dropout_p=0.3, kernel_size=3, equiv_mode=False):

        super().__init__()

        self.equiv_mode = equiv_mode

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

        # equiv:
        self.conv1 = nn.Conv2d(in_ch, base, kernel_size=kernel_size, padding='same')
        self.bn1 = nn.BatchNorm2d(base)

        self.conv2 = nn.Conv2d(base, base * 2, kernel_size=kernel_size, padding='same')
        self.bn2 = nn.BatchNorm2d(base * 2)

        self.conv3 = nn.Conv2d(base * 2, base * 4, kernel_size=kernel_size, padding='same')
        self.bn3 = nn.BatchNorm2d(base * 4)

        self.conv4 = nn.Conv2d(base * 4, base * 8, kernel_size=kernel_size, padding='same')
        self.bn4 = nn.BatchNorm2d(base * 8)

    def forward(self, x):

        if (self.equiv_mode):

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