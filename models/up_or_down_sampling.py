import torch.nn as nn
import torch
import torch.nn.functional as F


def get_weight(module,
               shape,
               weight_var='weight',
               kernel_init=None):
  """Get/create weight tensor for a convolution or fully-connected layer."""

  return module.param(weight_var, kernel_init, shape)


def naive_upsample_2d(x, factor=2):
  N, C, H, W = x.shape
  x = F.interpolate(x, (H * factor, W * factor), mode='nearest')
  return x


def naive_downsample_2d(x, factor=2):
  x = F.avg_pool2d(x, kernel_size=(factor, factor), stride=(factor, factor), padding=0)
  return x


