"""
PCME
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import numpy as np

import torch
import torch.nn.functional as F


def to_numpy(tensor, n_dims=2):
    """Convert a torch tensor to numpy array.

    Args:
        tensor (Tensor): a tensor object to convert.
        n_dims (int): size of numpy array shape
    """
    try:
        nparray = tensor.detach().cpu().clone().numpy()
    except AttributeError:
        raise TypeError('tensor type should be torch.Tensor, not {}'.format(type(tensor)))

    while len(nparray.shape) < n_dims:
        nparray = np.expand_dims(nparray, axis=0)

    return nparray


def l2_normalize(tensor, axis=-1):
    """L2-normalize columns of tensor"""
    return F.normalize(tensor, p=2, dim=axis)


def sample_gaussian_tensors(mu, logsigma, num_samples):
    eps = torch.randn(mu.size(0), num_samples, mu.size(1), dtype=mu.dtype, device=mu.device)

    samples = eps.mul(torch.exp(logsigma.unsqueeze(1))).add_(
        mu.unsqueeze(1))
    return samples
