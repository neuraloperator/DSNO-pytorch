import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import layers
from . import up_or_down_sampling

default_init = layers.default_init
conv3x3 = layers.ddpm_conv3x3
conv1x1 = layers.ddpm_conv1x1
NIN = layers.NIN


@torch.jit.script
def _scale(h, scale, shift):
    return h * (scale + 1.0) + shift


class ResidualBlockm(nn.Module):
    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, up=False, down=False,
                 dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1), init_scale=0.):
        super().__init__()

        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6)
        self.up = up
        self.down = down
        self.fir = fir
        self.fir_kernel = fir_kernel

        self.Conv_0 = conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, 2 * out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
            nn.init.zeros_(self.Dense_0.bias)

        self.GroupNorm_1 = nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch:
            self.NIN_0 = NIN(in_ch, out_ch)

        self.act = act
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x, temb=None):
        # temb: [B, T, C]
        B, T, C, H, W = x.shape
        xview = x.reshape(B * T, C, H, W)
            # B, C, H, W
        h = self.act(self.GroupNorm_0(xview))

        if self.up:
                h = up_or_down_sampling.naive_upsample_2d(h, factor=2)
                xview = up_or_down_sampling.naive_upsample_2d(xview, factor=2)
        elif self.down:
                h = up_or_down_sampling.naive_downsample_2d(h, factor=2)
                xview = up_or_down_sampling.naive_downsample_2d(xview, factor=2)

        h = self.Conv_0(h)  # B, Cout, H, W
        # Add bias to each feature map conditioned on the time embedding
        if temb is not None:
            temb_out = self.Dense_0(self.act(temb))[:, :, :, None, None]
            temb_out = temb_out.reshape(B * T, 2 * self.out_ch, 1, 1)
            # T, Cout, 1, 1
            scale, shift = temb_out.chunk(2, dim=1)
            # scale: T, Cout, 1, 1
            h = self.GroupNorm_1(h)
            # N, C, H, W
            h = self.act(_scale(h, scale, shift))
        else:
            h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)

        if self.in_ch != self.out_ch:
            xview = self.NIN_0(xview)

        N, C, H, W = xview.shape
        h = h.reshape(B, T, self.out_ch, H, W)

        xview = xview.reshape(B, -1, C, H, W)
        
        out = xview + h
        # print(out.shape)
        return out


def scaled_dot_product(q, k, v):
    '''
    Input:
        (B, num_head, len_seq, head_dim)
    Output:
        attention: (B, num_head, len_seq, len_seq)
        values: (B, num_head, len_seq, head_dim)
    '''
    dim = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(dim)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values


class MultiheadAttnBlock(nn.Module):
    def __init__(self, channels, num_heads, head_dim=None):
        """
            Multihead attention block.
        Args:
            channels (int): number of channels
            num_heads (int): number of attn heads
            head_dim (int, optional): head dimension, if specified, it will overwrite the number of heads. 
        """
        super(MultiheadAttnBlock, self).__init__()
        if head_dim is None:
            assert channels % num_heads == 0
            self.head_dim = channels // num_heads
            self.num_heads = num_heads
        else:
            assert channels % head_dim == 0
            self.head_dim = head_dim
            self.num_heads = channels // head_dim
        
        self.ch = channels
        self.NINs = NIN(channels, 3 * channels)
        self.NIN_3 = NIN(channels, channels, init_scale=0.0)
        self.GroupNorm_0 = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)

    def forward(self, x):
        B, T, C, H, W = x.shape
        N = B * T
        xview = x.reshape(N, C, H, W)

        h = self.GroupNorm_0(xview)

        h = self.NINs(h)    # N, 3C, H, W
        h = h.reshape(N, self.num_heads, 3 * self.head_dim, H * W)
        h = h.permute(0, 1, 3, 2)       # (N, num_heads, H * W, 3 * head_dim)
        q, k, v = h.chunk(3, dim=-1)     # each chunk has shape (B, num_heads, H * W, head_dim)
        h = scaled_dot_product(q, k, v) # (N, num_heads, H * W, head_dim)
        h = h.permute(0, 1, 3, 2)       # (N, num_heads, head_dim, H * W)
        h = h.reshape(N, self.num_heads * self.head_dim, H, W)
        h = self.NIN_3(h)
        return x + h.reshape(B, T, C, H, W)