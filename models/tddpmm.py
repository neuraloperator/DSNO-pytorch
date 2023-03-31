import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from . import layers
from . import layersmt

get_act = layers.get_act
default_initializer = layers.default_init
conv3x3 = layers.ddpm_conv3x3
Downsample = layers.Downsample
Upsample = layers.Upsample
NIN = layers.NIN
time_conv = layers.TimeConv
MultiheadAttn = layersmt.MultiheadAttnBlock
ResidualBlockm = layersmt.ResidualBlockm


def get_logsnr_schedule(logsnr_max=20.0, logsnr_min=-20.0):
    b = np.arctan(np.exp(- 0.5 * logsnr_max))
    a = np.arctan(np.exp(- 0.5 * logsnr_min)) - b
    def get_logsnr(t):
        out = - 2.0 * torch.log(torch.tan(a * t + b))
        return out
    return get_logsnr


def get_logsnr_input(logsnr, logsnr_type='inv_cos'):
    if logsnr_type == 'inv_cos':
        logsnr_input = torch.atan(torch.exp(- 0.5 * torch.clamp(logsnr, min=-20., max=20.))) / (0.5 * np.pi)
    elif logsnr_type == 'sigmoid':
        logsnr_input = torch.sigmoid(logsnr)
    else:
        raise ValueError(f'{logsnr_type} not supported')
    return logsnr_input


def get_timestep_embedding(timesteps, embedding_dim, max_time=1000.):
    timesteps *= (1000.0 / max_time)
    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(- torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb



class TDDPMm(nn.Module):
    def __init__(self, config):
        super(TDDPMm, self).__init__()
        self.config = config
        self.act = act = get_act(config)
        self.nf = nf = config.model.nf
        self.temb_dim = temb_dim = config.model.temb_dim
        ch_mult = config.model.ch_mult
        self.use_time_conv = use_time_conv = config.model.time_conv
        self.with_nin = with_nin = config.model.with_nin
        self.num_modes = num_modes = config.model.num_modes
        self.pred_eps = config.model.pred_eps

        self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
        self.attn_resolutions = attn_resolutions = config.model.attn_resolutions

        self.num_attn_heads = num_attn_heads = config.model.num_attn_heads if hasattr(config.model, 'num_attn_heads') else None
        self.head_dim = head_dim = config.model.head_dim if hasattr(config.model, 'head_dim') else None

        self.resblock_type = resblock_type = config.model.resblock_type.lower()
        dropout = config.model.dropout
        resamp_with_conv = config.model.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [config.data.image_size // (2 ** i) for i in range(num_resolutions)]

        self.conditional = conditional = config.model.conditional  # noise-conditional
        self.num_classes = num_classes = config.model.num_classes
        self.logsnr_type = config.model.logsnr_type        
        
        channels = config.data.num_channels
        
        self.fourier_feature = config.model.fourier_feature
        if self.fourier_feature:
            in_channels = channels + 3
        else:
            in_channels = channels
        
        init_scale = config.model.init_scale

        ResnetBlock = partial(ResidualBlockm,
                              act=act,
                              dropout=dropout,
                              init_scale=init_scale,
                              temb_dim=temb_dim)
        AttnBlock = partial(MultiheadAttn,
                            num_heads=num_attn_heads, head_dim=head_dim)

        if conditional:
            # Condition on noise levels.
            modules = [nn.Linear(nf, temb_dim)]
            modules[0].weight.data = default_initializer()(modules[0].weight.data.shape)
            nn.init.zeros_(modules[0].bias)
            modules.append(nn.Linear(temb_dim, temb_dim))
            modules[1].weight.data = default_initializer()(modules[1].weight.data.shape)
            nn.init.zeros_(modules[1].bias)
        if num_classes > 0:
            modules.append(nn.Linear(num_classes, temb_dim))

        modules.append(conv3x3(in_channels, nf))
        hs_c = [nf]
        in_ch = nf
        # downsample part
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch
                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                if use_time_conv:
                    modules.append(time_conv(in_ch=in_ch, out_ch=out_ch, modes=num_modes, act=act, with_nin=with_nin))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                if resblock_type == 'ddpm':
                    modules.append(Downsample(in_ch, with_conv=resamp_with_conv))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, down=True))
                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        if use_time_conv:
            modules.append(time_conv(in_ch=in_ch, out_ch=in_ch, modes=num_modes, act=act, with_nin=with_nin))
        modules.append(ResnetBlock(in_ch=in_ch))

        # upsample part
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch
                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                if use_time_conv:
                    modules.append(time_conv(in_ch=in_ch, out_ch=in_ch, modes=num_modes, act=act, with_nin=with_nin))

            if i_level != 0:
                if resblock_type == 'ddpm':
                    modules.append(Upsample(channels=in_ch, with_conv=resamp_with_conv))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))
        assert not hs_c
        modules.append(nn.GroupNorm(num_channels=in_ch, num_groups=32, eps=1e-6))
        modules.append(conv3x3(in_ch, channels, init_scale=0.))
        self.all_modules = nn.ModuleList(modules)

    def forward(self, x, logsnr, y=None):
        B = x.shape[0]
        T = logsnr.shape[0]

        modules = self.all_modules
        m_idx = 0
        if self.conditional:
            # timestep/scale embedding
            logsnr_input = get_logsnr_input(logsnr, logsnr_type=self.logsnr_type)
            temb = get_timestep_embedding(logsnr_input, self.nf, max_time=1.0)
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None
        if y is not None:
            onehot = F.one_hot(y, num_classes=self.num_classes).to(x.dtype)
            emb = modules[m_idx](onehot)
            m_idx += 1
            temb = temb[None, :, :] + emb[:, None, :]   # B, T, C

        if len(temb.shape) < 3:
            temb = temb[None, :, :].repeat_interleave(B, dim=0)

        if self.pred_eps:
            noise = x[:, None, :, :, :]

        if self.fourier_feature:
            hf = base2FourierFeatures(x, start=6, stop=8, step=1)
            h = torch.cat([x, hf], dim=1)
        else:
            h = x
        # Downsampling block
        h0 = modules[m_idx](h)[:, None, :, :, :].repeat_interleave(T, dim=1)      # B, T, C, H, W
        hs = [h0]
        m_idx += 1
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1
                if self.use_time_conv:
                    h = modules[m_idx](h)
                    m_idx += 1
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                h = modules[m_idx](hs[-1], temb)
                hs.append(h)
                m_idx += 1

        h = hs[-1]
        h = modules[m_idx](h, temb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        if self.use_time_conv:
            h = modules[m_idx](h)
            m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=2), temb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1
                if self.use_time_conv:
                    h = modules[m_idx](h)
                    m_idx += 1
            if i_level != 0:
                h = modules[m_idx](h, temb)
                m_idx += 1

        assert not hs
        B, T, C, H, W = h.shape

        h = self.act(modules[m_idx](h.view(B * T, C, H, W)))
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        assert m_idx == len(modules)
        h = h.view(B, T, 3, H, W)
        if self.pred_eps:
            h = noise - h
        return h


# class Base2FourierFeatures(nn.Modules):
#     def __init__(self, start=0, stop=8, step=1) -> None:
#         super().__init__()

def base2FourierFeatures(x, start=0, stop=8, step=1):
    freqs = torch.range(start, stop, step, dtype=x.dtype)
    w = 2. ** freqs * 2 * np.pi
    w = torch.tile(w[None, :], (1, x.shape[-1]))

    h = x.repeat_interleave(freqs.shape[0], dim=-1)
    h = w * h
    h = torch.cat([torch.sin(h), torch.cos(h)], dim=-1)
    return h