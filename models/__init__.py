""" Model wrapper for PCME

PCME
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
from models.pcme import PCME


__all__ = ['get_model']


def get_model(model_name, word2idx, config):
    if model_name == 'pcme':
        return PCME(word2idx, config)
    else:
        raise ValueError(f'Invalid model name: {model_name}')
