"""
PCME
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import pickle
import yaml
from yaml.error import YAMLError

import pandas as pd
import torch


def flatten_dict(dict_, sep):
    return pd.json_normalize(dict_, sep=sep).to_dict(orient='records')[0]


def object_loader(config_path):
    with open(config_path, 'r') as fin:
        try:
            return yaml.load(fin)
        except YAMLError:
            print('failed to load from yaml. Try pickle loader')
        try:
            return pickle.load(fin)
        except TypeError:
            print('failed to load from pickle. Try torch loader')
        try:
            return torch.load(fin)
        except TypeError:
            print('failed to load from pickle. Please check your configuration again.')
    raise TypeError('config_path should be serialized by [yaml, json, pickle, torch pth]')


def torch_safe_load(module, state_dict, strict=True):
    module.load_state_dict({
        k.replace('module.', ''): v for k, v in state_dict.items()
    }, strict=strict)
