import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialConv2d(nn.Module):
    
    #y = conv(x * m) / conv(m) (renormalized by number of valid inputs)

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding='same', bias=True):

        super().__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=bias)
        # buffer to count valid inputs (ones over input channels)
        k = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

        self.register_buffer('ones_weight', torch.ones(1, in_ch, *k))
        self.kernel_size = k
        self.stride = stride
        self.padding = padding
        self.in_ch = in_ch
        self.out_ch = out_ch

    # x: [N, C_in, E, A], mask: [N, 1, E, A]
    def forward(self, x, mask):

        # multiply by mask (broadcast over channels)
        xm = x * mask

        # numerator: normal conv on masked input
        num = self.conv(xm)

        # denominator: number of valid inputs per location in RF (convolve mask with ones over in_ch)
        # build conv2d over mask with the same stride/padding, then broadcast to out_ch
        den = F.conv2d(mask.repeat(1, self.in_ch, 1, 1),  # [N, C_in, E, A]
                       self.ones_weight,               # [1, C_in, kW, kH]
                       bias=None, stride=self.stride, padding=self.padding)
        
        den = den.clamp_min(1e-8)
        y = num / den

        # output mask: any valid input in RF -> 1
        m_out = (F.conv2d(mask, torch.ones(1, 1, *self.kernel_size, device=mask.device, dtype=mask.dtype),
                          bias=None, stride=self.stride, padding=self.padding) > 0).float()
        
        return y, m_out

# masked global average pool
def masked_gap(x, mask, eps=1e-8):

    w = mask.expand_as(x)
    num = (x * w).sum(dim=(2, 3))
    den = w.sum(dim=(2, 3)).clamp_min(eps)

    return num / den


class ResonanceCNN_Masked(nn.Module):

    def __init__(self, in_ch=2, base=80, dropout_p=0.3, kernel_size=3):

        super().__init__()

        self.p1 = PartialConv2d(in_ch=1, out_ch=base, kernel_size=kernel_size, padding='same')
        self.gn1 = nn.GroupNorm(8, base)
        self.p2 = PartialConv2d(in_ch=base, out_ch=base*2, kernel_size=kernel_size, padding='same')
        self.gn2 = nn.GroupNorm(8, base*2)
        self.p3 = PartialConv2d(in_ch=base*2, out_ch=base*4, kernel_size=kernel_size, padding='same')
        self.gn3 = nn.GroupNorm(8, base*4)
        self.p4 = PartialConv2d(in_ch=base*4, out_ch=base*8, kernel_size=kernel_size, padding='same')
        self.gn4 = nn.GroupNorm(8, base*8)

        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.fc1 = nn.Linear(base*8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(dropout_p)
        self.head_E = nn.Linear(128, 1)
        self.head_G = nn.Linear(128, 1)

    def forward(self, x):

        x = x[:, :1, :, :]
        mask = x[:, -1:, :, :].clamp(0,1)

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

        feat = masked_gap(x, mask)

        z = F.relu(self.fc1(feat))
        z = self.dropout(F.relu(self.fc2(z)))

        Er_unit  = torch.sigmoid(self.head_E(z)).squeeze(-1)
        logGamma = self.head_G(z).squeeze(-1)

        return Er_unit, logGamma