import math
from functools import partial
import numpy as np
import string

import torch.nn as nn
import torch
import torch.nn.functional as F


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    # assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    # emb = math.log(2.) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
    # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
    """Ported from JAX. """

    def _compute_fans(shape, in_axis=1, out_axis=0):
        receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out

    def init(shape, dtype=dtype, device=device):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(
                "invalid mode for variance scaling initializer: {}".format(mode))
        variance = scale / denominator
        if distribution == "normal":
            return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
        elif distribution == "uniform":
            return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
        else:
            raise ValueError("invalid distribution for variance scaling initializer")

    return init


def default_init(scale=1.):
    """The same initialization used in DDPM."""
    scale = 1e-10 if scale == 0 else scale
    return variance_scaling(scale, 'fan_avg', 'uniform')


def get_act(config):
    """Get activation functions from the config file."""

    if config.model.nonlinearity.lower() == 'elu':
        return nn.ELU()
    elif config.model.nonlinearity.lower() == 'relu':
        return nn.ReLU()
    elif config.model.nonlinearity.lower() == 'lrelu':
        return nn.LeakyReLU(negative_slope=0.2)
    elif config.model.nonlinearity.lower() == 'swish':
        return nn.SiLU()
    else:
        raise NotImplementedError('activation function does not exist!')


def ncsn_conv1x1(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1., padding=0):
    """1x1 convolution. Same as NCSNv1/v2."""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias, dilation=dilation,
                     padding=padding)
    init_scale = 1e-10 if init_scale == 0 else init_scale
    conv.weight.data *= init_scale
    conv.bias.data *= init_scale
    return conv


def ddpm_conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1., padding=1):
    """3x3 convolution with DDPM initialization."""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding,
                     dilation=dilation, bias=bias)
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv


def ddpm_conv1x1x1(in_planes, out_planes, stride=1, bias=True, init_scale=1., padding=0):
    """1x1x1 convolution with DDPM initialization."""
    conv = nn.Conv3d(in_planes, out_planes,
                     kernel_size=1,
                     stride=stride, padding=padding, bias=bias)
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv


def ddpm_conv1x1(in_planes, out_planes, stride=1, bias=True, init_scale=1., padding=0):
    """1x1 convolution with DDPM initialization."""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=bias)
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv


def ddpm_conv1x3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1., padding=1):
    """1x3x3 convolution with DDPM initialization."""
    conv = nn.Conv3d(in_planes, out_planes,
                     kernel_size=(1, 3, 3),
                     stride=stride,
                     padding=(0, padding, padding),
                     dilation=dilation, bias=bias)
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv


def ncsn_conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1., padding=1):
    """3x3 convolution with PyTorch initialization. Same as NCSNv1/NCSNv2."""
    init_scale = 1e-10 if init_scale == 0 else init_scale
    conv = nn.Conv2d(in_planes, out_planes, stride=stride, bias=bias,
                     dilation=dilation, padding=padding, kernel_size=3)
    conv.weight.data *= init_scale
    conv.bias.data *= init_scale
    return conv


class ConvMeanPool(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True, adjust_padding=False):
        super().__init__()
        if not adjust_padding:
            conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
            self.conv = conv
        else:
            conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)

            self.conv = nn.Sequential(
                nn.ZeroPad2d((1, 0, 1, 0)),
                conv
            )

    def forward(self, inputs):
        output = self.conv(inputs)
        output = sum([output[:, :, ::2, ::2], output[:, :, 1::2, ::2],
                      output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
        return output


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, resample=None, act=nn.ELU(),
                 normalization=nn.InstanceNorm2d, adjust_padding=False, dilation=1):
        super().__init__()
        self.non_linearity = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        self.normalization = normalization
        if resample == 'down':
            if dilation > 1:
                self.conv1 = ncsn_conv3x3(input_dim, input_dim, dilation=dilation)
                self.normalize2 = normalization(input_dim)
                self.conv2 = ncsn_conv3x3(input_dim, output_dim, dilation=dilation)
                conv_shortcut = partial(ncsn_conv3x3, dilation=dilation)
            else:
                self.conv1 = ncsn_conv3x3(input_dim, input_dim)
                self.normalize2 = normalization(input_dim)
                self.conv2 = ConvMeanPool(input_dim, output_dim, 3, adjust_padding=adjust_padding)
                conv_shortcut = partial(ConvMeanPool, kernel_size=1, adjust_padding=adjust_padding)

        elif resample is None:
            if dilation > 1:
                conv_shortcut = partial(ncsn_conv3x3, dilation=dilation)
                self.conv1 = ncsn_conv3x3(input_dim, output_dim, dilation=dilation)
                self.normalize2 = normalization(output_dim)
                self.conv2 = ncsn_conv3x3(output_dim, output_dim, dilation=dilation)
            else:
                # conv_shortcut = nn.Conv2d ### Something wierd here.
                conv_shortcut = partial(ncsn_conv1x1)
                self.conv1 = ncsn_conv3x3(input_dim, output_dim)
                self.normalize2 = normalization(output_dim)
                self.conv2 = ncsn_conv3x3(output_dim, output_dim)
        else:
            raise Exception('invalid resample value')

        if output_dim != input_dim or resample is not None:
            self.shortcut = conv_shortcut(input_dim, output_dim)

        self.normalize1 = normalization(input_dim)

    def forward(self, x):
        output = self.normalize1(x)
        output = self.non_linearity(output)
        output = self.conv1(output)
        output = self.normalize2(output)
        output = self.non_linearity(output)
        output = self.conv2(output)

        if self.output_dim == self.input_dim and self.resample is None:
            shortcut = x
        else:
            shortcut = self.shortcut(x)

        return shortcut + output


def _einsum(a, b, c, x, y):
    einsum_str = '{},{}->{}'.format(''.join(a), ''.join(b), ''.join(c))
    return torch.einsum(einsum_str, x, y)


def contract_inner(x, y):
    """tensordot(x, y, 1)."""
    x_chars = list(string.ascii_lowercase[:len(x.shape)])
    y_chars = list(string.ascii_lowercase[len(x.shape):len(y.shape) + len(x.shape)])
    y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
    out_chars = x_chars[:-1] + y_chars[1:]
    return _einsum(x_chars, y_chars, out_chars, x, y)


class NIN(nn.Module):
    def __init__(self, in_dim, num_units, init_scale=0.1):
        super().__init__()
        self.W = nn.Parameter(default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

    def forward(self, x):
        # x: (B, C, H, W)
        y = torch.einsum('bchw,co->bohw', x, self.W) + self.b[None, :, None, None]
        if y.stride()[1] == 1:
            y = y.contiguous()
        return y


class AttnBlock(nn.Module):
    """Channel-wise self-attention block."""

    def __init__(self, channels):
        super().__init__()
        self.GroupNorm_0 = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)
        self.NIN_0 = NIN(channels, channels)
        self.NIN_1 = NIN(channels, channels)
        self.NIN_2 = NIN(channels, channels)
        self.NIN_3 = NIN(channels, channels, init_scale=0.)

    def forward(self, x):
        B, C, T, H, W = x.shape
        h = self.GroupNorm_0(x)
        q = self.NIN_0(h)
        k = self.NIN_1(h)
        v = self.NIN_2(h)

        w = torch.einsum('bcthw,bctij->bthwij', q, k) * (int(C) ** (-0.5))
        w = torch.reshape(w, (B, T, H, W, H * W))
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, (B, T, H, W, H, W))
        h = torch.einsum('bthwij,bctij->bcthw', w, v)
        h = self.NIN_3(h)
        return x + h


class Upsample(nn.Module):
    def __init__(self, channels, with_conv=False):
        super().__init__()
        if with_conv:
            self.Conv_0 = ddpm_conv1x3x3(channels, channels)
        self.with_conv = with_conv

    def forward(self, x):
        B, C, T, H, W = x.shape
        h = F.interpolate(x, (T, H * 2, W * 2), mode='nearest')
        if self.with_conv:
            h = self.Conv_0(h)
        return h


class Downsample(nn.Module):
    def __init__(self, channels, with_conv=False):
        super().__init__()
        if with_conv:
            self.Conv_0 = ddpm_conv1x3x3(channels, channels, stride=(1, 2, 2), padding=0)
        self.with_conv = with_conv

    def forward(self, x):
        # Emulate 'SAME' padding
        if self.with_conv:
            x = F.pad(x, (0, 1, 0, 1))
            x = self.Conv_0(x)
        else:
            x = F.avg_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)
        return x


class ResnetBlockDDPM(nn.Module):
    """Adapted from the ResNet Blocks used in DDPM
    """

    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, conv_shortcut=False, dropout=0.1):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        self.GroupNorm_0 = nn.GroupNorm(num_groups=32, num_channels=in_ch, eps=1e-6)
        self.act = act
        self.Conv_0 = ddpm_conv1x3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
            nn.init.zeros_(self.Dense_0.bias)

        self.GroupNorm_1 = nn.GroupNorm(num_groups=32, num_channels=out_ch, eps=1e-6)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = ddpm_conv1x3x3(out_ch, out_ch, init_scale=0.)
        if in_ch != out_ch:
            if conv_shortcut:
                self.Conv_2 = ddpm_conv1x3x3(in_ch, out_ch)
            else:
                self.NIN_0 = NIN(in_ch, out_ch)
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.conv_shortcut = conv_shortcut

    def forward(self, x, temb=None):
        B, C, T, H, W = x.shape
        assert C == self.in_ch
        out_ch = self.out_ch if self.out_ch else self.in_ch
        h = self.act(self.GroupNorm_0(x))
        h = self.Conv_0(h)
        # Add bias to each feature map conditioned on the time embedding
        # temb: (T, C_in)
        if temb is not None:
            h += self.Dense_0(self.act(temb)).permute(1, 0)[None, :, :, None, None]
        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        if C != out_ch:
            if self.conv_shortcut:
                x = self.Conv_2(x)
            else:
                x = self.NIN_0(x)
        return x + h


class BBlock(nn.Module):
    '''
    Block for lifting channels
    project and add
    '''
    def __init__(self, in_ch, out_ch, act, temb_dim=None):
        super(BBlock, self).__init__()
        self.proj = ddpm_conv3x3(in_ch, out_ch)
        self.act = act
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init(1e-3)(self.Dense_0.weight.data.shape)
            nn.init.zeros_(self.Dense_0.bias)

    def forward(self, x, temb=None):
        '''
        Args:
            x: (B, C, H, W)
            temb: (T, C_out)

        return:
            out: (B, T, C, H, W)
        '''
        h = self.proj(x)[:, None, :, :, :]
        if temb is not None:
            T = temb.shape[0]
            out = h.repeat_interleave(T, dim=1)
        else:
            out = h
        # if temb is not None:
        #     # out_temb = self.Dense_0(self.act(temb))
        #     th = torch.zeros(h.shape[1], temb.shape[0], device=temb.device)[None, :, :, None, None]
        # else:
        #     th = 0
        return out

@torch.jit.script
def compl_mul1d(a, b):
    # (B, M, in_ch, H, W), (in_ch, out_ch, M) -> (B, M, out_channel, H, W)
    return torch.einsum("bmihw,iom->bmohw", a, b)


class SpectralConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes1 = modes1

        self.scale = (1 / (in_ch*out_ch))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_ch, out_ch, self.modes1, 2, dtype=torch.float))


    def forward(self, x):
        B, T, C, H, W = x.shape
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        with torch.autocast(device_type='cuda', enabled=False):
            x_ft = torch.fft.rfftn(x.float(), dim=[1])
            # Multiply relevant Fourier modes
            out_ft = compl_mul1d(x_ft[:, :self.modes1], torch.view_as_complex(self.weights1))
            # Return to physical space
            x = torch.fft.irfftn(out_ft, s=[T], dim=[1])
        return x


class TimeConv(nn.Module):
    def __init__(self, in_ch, out_ch, modes, act, with_nin=False):
        super(TimeConv, self).__init__()
        self.with_nin = with_nin
        self.t_conv = SpectralConv1d(in_ch, out_ch, modes)
        if with_nin:
            self.nin = NIN(in_ch, out_ch)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        h = self.t_conv(x)
        if self.with_nin:
            x = self.nin(x)
        out = self.act(h)
        return x + out