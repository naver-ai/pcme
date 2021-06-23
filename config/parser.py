"""
PCME
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import munch

from utils.serialize_utils import object_loader


def parse_config(config_path, cache_dir=None, pretrained_resnet_model_path=None, use_fp16=False):
    dict_config = object_loader(config_path)

    config = {}
    for config_key, subconfig in dict_config.items():
        if not isinstance(subconfig, dict):
            raise TypeError('unexpected type Key({}) Value({}) '
                            'All({})'.format(config_key, subconfig, config))

        for subconfig_key, subconfig_value in subconfig.items():
            if isinstance(subconfig_value, dict):
                raise ValueError('Only support two-depth configs. '
                                 'See README. All({})'.format(config))

        config[config_key] = munch.Munch(**subconfig)

    config = munch.Munch(**config)
    config.train.use_fp16 = use_fp16
    config.model.cache_dir = cache_dir
    config.model.pretrained_resnet_model_path = pretrained_resnet_model_path
    return config


def config_to_dict(config):
    return {k: dict(v) for k, v in config.items()}
