"""Criterions for cross-modal retrieval methods.
This module contains the following criterions:
- MC soft contrastive loss for PCME

PCME
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
from criterions.probemb import MCSoftContrastiveLoss


def get_criterion(criterion_name, config):
    if criterion_name == 'pcme':
        return MCSoftContrastiveLoss(config)
    else:
        raise ValueError(f'Invalid criterion name: {criterion_name}')
