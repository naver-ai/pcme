"""Batch-wise efficient probabilistic embedding loss for cross-modal retrieval

PCME
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import numpy as np

import torch
import torch.nn as nn


def batchwise_cdist(samples1, samples2, eps=1e-6):
    """Compute L2 distance between each pair of the two multi-head embeddings in batch-wise.
    We may assume that samples have shape N x K x D, N: batch_size, K: number of embeddings, D: dimension of embeddings.
    The size of samples1 and samples2 (`N`) should be either
    - same (each sample-wise distance will be computed separately)
    - len(samples1) = 1 (samples1 will be broadcasted into samples2)
    - len(samples2) = 1 (samples2 will be broadcasted into samples1)

    The following broadcasting operation will be computed:
    (N x 1 x K x D) - (N x K x 1 x D) = (N x K x K x D)

    Parameters
    ----------
    samples1: torch.Tensor (shape: N x K x D)
    samples2: torch.Tensor (shape: N x K x D)

    Returns
    -------
    batchwise distance: N x K ** 2
    """
    if len(samples1.size()) != 3 or len(samples2.size()) != 3:
        raise RuntimeError('expected: 3-dim tensors, got: {}, {}'.format(samples1.size(), samples2.size()))

    if samples1.size(0) == samples2.size(0):
        batch_size = samples1.size(0)
    elif samples1.size(0) == 1:
        batch_size = samples2.size(0)
    elif samples2.size(0) == 1:
        batch_size = samples1.size(0)
    else:
        raise RuntimeError(f'samples1 ({samples1.size()}) and samples2 ({samples2.size()}) dimensionalities '
                           'are non-broadcastable.')

    samples1 = samples1.unsqueeze(1)
    samples2 = samples2.unsqueeze(2)
    return torch.sqrt(((samples1 - samples2) ** 2).sum(-1) + eps).view(batch_size, -1)


def soft_contrastive_nll(logit, matched):
    r"""Compute the negative log-likelihood of the soft contrastive loss.

    .. math::
        NLL_{ij} = -\log p(m = m_{ij} | z_i, z_j)
                 = -\log \left[ \mathbb{I}_{m_{ij} = 1} \sigma(-a \| z_i - z_j \|_2 + b)
                         +  \mathbb{I}_{m_{ij} = -1} (1 - \sigma(-a \| z_i - z_j \|_2 + b)) \right].

    Note that the matching indicator {m_ij} is 1 if i and j are matched otherwise -1.
    Here we define the sigmoid function as the following:
    .. math::
        \sigma(x) = \frac{\exp(x)}{\exp(x) + \exp(-x)}, \text{ i.e., }
        1 - \sigma(x) = \frac{\exp(-x)}{\exp(x) + \exp(-x)}.

    Here we sample "logit", s_{ij} by Monte-Carlo sampling to get the expected soft contrastive loss.
    .. math::
        s_{ij}^k = -a \| z_i^k - z_j^k \|_2 + b, z_i^k ~ \mathcal N (\mu_i, \Sigma_i), z_j^k ~ \mathcal N (\mu_j, \Sigma_j).

    Then we can compute NLL by logsumexp (here, we omit `k` in s_{ij}^k for the simplicity):
    .. math::
        NLL_{ij} = -\log \left[ \frac{1}{K^2} \sum_{s_{ij}} \left{ \frac{\exp(s_{ij} m_ij)}{\exp(s_{ij}) + \exp(-s_{ij})} \right} \right]
                 = (\log K^2) -\log \sum_{s_{ij}} \left[ \exp \left( s_{ij} m_ij - \log(\exp(s_{ij} + (-s_{ij}))) \right) \right]
                 = (\log K^2) -logsumexp( s_{ij} m_{ij} - logsumexp(s_{ij}, -s_{ij}) ).

    Parameters
    ----------
    logit: torch.Tensor (shape: N x K ** 2)
    matched: torch.Tensor (shape: N), an element should be either 1 (matched) or -1 (mismatched)

    Returns
    -------
    NLL loss: torch.Tensor (shape: N), should apply `reduction` operator for the backward operation.
    """
    if len(matched.size()) == 1:
        matched = matched[:, None]
    return -(
        (logit * matched - torch.stack(
            (logit, -logit), dim=2).logsumexp(dim=2, keepdim=False)
         ).logsumexp(dim=1)) + np.log(logit.size(1))


class MCSoftContrastiveLoss(nn.Module):
    r"""Creates a criterion that measures the pairwise soft contrastive loss given
    input tensor pairs :math:`X`, :math:`Y` where each tensor is already sampled from a distribution.

    .. math::
        \log p(m = \hat m | x, y)
        p(m = 1 | x, y) = \sigma(-a \| x - y \|_2 + b)
        p(m = 0 | x, y) = 1 - \sigma(-a \| x - y \|_2 + b)
        \sigma(x) = \frac{\exp(x)}{\exp(x) + \exp(-x)}, \text{ i.e., }
        1 - \sigma(x) = \frac{\exp(-x)}{\exp(x) + \exp(-x)}.

    This code assumes that :math:`x_i` and :math:`y_j` are in same class if i = j,
    and in different class otherwise.

    The division by :math:`n` can be avoided if sets ``reduction = 'sum'``.

    Parameters
    ----------
    TBD

    Shape
    -----
    Input1 : torch.Tensor
        :math:`(N, K, D)` shape, `N` is the batch size, `K` is the number of samples and `D` is the size of a sample.
    Input2: torch.Tensor
        :math:`(N, K, D)` shape, `N` is the batch size, `K` is the number of samples and `D` is the size of a sample.
    Output: torch.Tensor
        If :attr:`reduction` is ``'none'``, then :math:`(N)`.
    """
    def __init__(self, config, reduction='sum'):
        super().__init__()
        if reduction not in {'mean', 'sum', None}:
            raise ValueError('unknown reduction {}'.format(reduction))
        self.reduction = reduction

        shift = config.init_shift * torch.ones(1)
        negative_scale = config.init_negative_scale * torch.ones(1)

        shift = nn.Parameter(shift)
        negative_scale = nn.Parameter(negative_scale)

        self.register_parameter('shift', shift)
        self.register_parameter('negative_scale', negative_scale)

        self.num_samples = config.num_samples

        self.uniform_lambda = config.get('uniform_lambda', 0)
        self.vib_beta = config.get('vib_beta', 0)

    def uniform_loss(self, x, max_samples=16384, t=2):
        if len(x) ** 2 > max_samples:
            # prevent CUDA error: https://github.com/pytorch/pytorch/issues/22313
            indices = np.random.choice(len(x), int(np.sqrt(max_samples)))
            x = x[indices]
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def kl_divergence(self, mu, logsigma):
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum()

    def pairwise_sampling(self, anchors, candidates):
        N = len(anchors)
        if len(anchors) != len(candidates):
            raise RuntimeError('# anchors ({}) != # candidates ({})'.format(anchors.shape, candidates.shape))
        anchor_idx, selected_idx, matched = self.full_sampling(N)

        anchor_idx = torch.from_numpy(np.array(anchor_idx)).long()
        selected_idx = torch.from_numpy(np.array(selected_idx)).long()
        matched = torch.from_numpy(np.array(matched)).float()

        anchor_idx = anchor_idx.to(anchors.device)
        selected_idx = selected_idx.to(anchors.device)
        matched = matched.to(anchors.device)

        anchors = anchors[anchor_idx]
        selected = candidates[selected_idx]

        cdist = batchwise_cdist(anchors, selected)

        return cdist, matched

    def full_sampling(self, N):
        candidates = []
        selected = []
        matched = []
        for i in range(N):
            for j in range(N):
                candidates.append(i)
                selected.append(j)
                if i == j:
                    matched.append(1)
                else:
                    matched.append(-1)
        return candidates, selected, matched

    def _compute_loss(self, input1, input2):
        """
        Shape
        -----
        Input1 : torch.Tensor
            :math:`(N, K, D)` shape, `N` is the batch size, `K` is the number of samples and `D` is the size of a sample.
        Input2: torch.Tensor
            :math:`(N, K, D)` shape, `N` is the batch size, `K` is the number of samples and `D` is the size of a sample.
        Output: torch.Tensor
            If :attr:`reduction` is ``'none'``, then :math:`(N)`.
        """
        distance, matched = self.pairwise_sampling(input1, input2)
        logits = -self.negative_scale * distance + self.shift

        idx = matched == 1
        loss_pos = soft_contrastive_nll(logits[idx], matched[idx]).sum()
        idx = matched != 1
        loss_neg = soft_contrastive_nll(logits[idx], matched[idx]).sum()

        return {
            'loss': loss_pos + loss_neg,
            'pos_loss': loss_pos,
            'neg_loss': loss_neg,
        }

    def match_prob(self, image_features, caption_features, image_logsigma, caption_logsigma, use_batchwise_cdist=True):
        sampled_image_features, sampled_caption_features = image_features, caption_features
        distance = batchwise_cdist(sampled_image_features, sampled_caption_features)

        distance = distance.to(self.negative_scale.device)
        distance = distance.float()
        logits = -self.negative_scale * distance + self.shift
        prob = torch.exp(logits) / (torch.exp(logits) + torch.exp(-logits))

        return prob.mean(axis=1)

    def forward(self, image_features, caption_features, image_logsigma, caption_logsigma, **kwargs):
        uniform_loss = 0
        uniform_loss_val = 0
        vib_loss = 0
        vib_loss_val = 0

        if self.uniform_lambda != 0:
            dim = image_features.size()[-1]
            uniform_loss = self.uniform_loss(torch.cat([image_features.view(-1, dim), caption_features.view(-1, dim)]))
            uniform_loss_val = uniform_loss.item()
        sampled_image_features, sampled_caption_features = image_features, caption_features

        if self.vib_beta != 0:
            vib_loss =\
                self.kl_divergence(image_features.mean(dim=1), image_logsigma) + self.kl_divergence(caption_features.mean(dim=1), caption_logsigma)
            vib_loss_val = vib_loss.item()

        i2t_loss = self._compute_loss(sampled_image_features, sampled_caption_features)
        t2i_loss = self._compute_loss(sampled_caption_features, sampled_image_features)
        loss = i2t_loss['loss'] + t2i_loss['loss'] + self.uniform_lambda * uniform_loss + self.vib_beta * vib_loss

        loss_dict = {'i2t_loss': i2t_loss['loss'].item(),
                     't2i_loss': t2i_loss['loss'].item(),
                     'i2t_pos_loss': i2t_loss['pos_loss'].item(),
                     'i2t_neg_loss': i2t_loss['neg_loss'].item(),
                     't2i_pos_loss': t2i_loss['pos_loss'].item(),
                     't2i_neg_loss': t2i_loss['neg_loss'].item(),
                     'uniform_loss': uniform_loss_val,
                     'vib_loss': vib_loss_val,
                     'shift': self.shift.item(),
                     'negative_scale': self.negative_scale.item(),
                     'loss': loss.item()}
        return loss, loss_dict
