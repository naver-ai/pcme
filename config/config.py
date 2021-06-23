"""Python wrapper for experiment configurations.

PCME
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import os
import pickle
import munch
try:
    import ujson as json
except ImportError:
    import json
import yaml
from yaml.error import YAMLError

try:
    import torch
except ImportError:
    pass


def _print(msg, verbose):
    """A simple print wrapper.
    """
    if verbose:
        print(msg)


def _loader(config_path: str,
            verbose: bool = False) -> dict:
    """A simple serializer loader.

    Examples
    --------
    >>> _loader('test.json')
    """
    with open(config_path, 'r') as fin:
        try:
            return yaml.load(fin)
        except YAMLError:
            _print('failed to load from yaml. Try pickle loader', verbose)

    with open(config_path, 'rb') as fin:
        try:
            return pickle.load(fin)
        except TypeError:
            _print('failed to load from pickle. Try torch loader', verbose)
        try:
            return torch.load(fin)
        except TypeError:
            _print('failed to load from pickle. Please check your configuration again.', verbose)

    raise TypeError('config_path should be serialized by [yaml, json, pickle, torch pth]')


def dump_config(config: munch.Munch,
                dump_to: str,
                overwrite: bool = False,
                serializer: str = 'json'):
    """Dump the configuration to the local file {dump_to}.

    Parameters
    ----------
    config: munch.Munch
        A configuration file defines the structure of the configuration.
        The file should be serialized by any of [yaml, json, pickle, torch].

    dump_to: str
        A destination path to dump the configuration.

    overwrite: bool, optional, default=False,
        If False, raise FileExistsError if `dump_to` already exists.

    serializer: str, optional, default='json',
        Format to dump. It should be in ["json", "yaml", "pickle", "torch"]

    Examples
    --------
    >>> dump_config(config, 'my_simple_config.json')
    """
    if serializer not in {'json', 'yaml', 'pickle', 'torch'}:
        raise ValueError(f'format should be in ["json", "yaml", "pickle", "torch"], not {serializer}')

    if os.path.exists(dump_to) and not overwrite:
        raise FileExistsError(dump_to)

    if serializer in ('pickle', 'torch'):
        mode = 'wb'
    else:
        mode = 'w'

    config = munch.unmunchify(config)

    with open(dump_to, mode) as fout:
        if serializer == 'json':
            json.dump(config, fout, indent=4, sort_keys=True)
        elif serializer == 'yaml':
            yaml.dump(config, fout)
        elif serializer == 'pickle':
            pickle.dump(config, fout)
        elif serializer == 'torch':
            torch.save(config, fout)


def parse_config(config_fname: str,
                 delimiter: str = '__',
                 strict_cast: bool = True,
                 verbose: bool = False,
                 **kwargs) -> munch.Munch:
    """Parse the given configuration file with additional options to overwrite.

    Parameters
    ----------
    config_fname: str
        A configuration file defines the structure of the configuration.
        The file should be serialized by any of [yaml, json, pickle, torch].

    delimiter: str, optional, default='__'
        A delimiter for the additional kwargs configuration.
        See kwargs for more information.

    strict_cast: bool, optional, default=True
        If True, the overwritten config values will be casted as the original type.

    verbose: bool, optional, default=False

    kwargs: optional
        If specified, overwrite the current configuration by the given keywords.
        For the multi-depth configuration, "__" is used for the default delimiter.
        The keys in kwargs should be already defined by config_fname (otherwise it will raise KeyError).
        Note that if `strict_cast` is True, the values in kwargs will be casted as the original type defined in the configuration file.

    Returns
    -------
    config: munch.Munch
        A configuration file, which provides attribute-style access.
        See `Munch <https://github.com/Infinidat/munch>`_ project for the details.

    Examples
    --------
    >>> # simple_config.json => {"opt1": {"opt2": 1}, "opt3": 0}
    >>> config = parse_config('simple_config.json')
    >>> print(config.opt1.opt2, config.opt3, type(config.opt1.opt2), type(config.opt3))
    2 1 <class 'int'> <class 'int'>

    >>> config = parse_config('simple_config.json', opt1__opt2=2, opt3=1)
    >>> print(config.opt1.opt2, config.opt3, type(config.opt1.opt2), type(config.opt3))
    2 1 <class 'int'> <class 'int'>

    >>> parse_config('test.json', **{'opt1__opt2': '2', 'opt3': 1.0})
    >>> print(config.opt1.opt2, config.opt3, type(config.opt1.opt2), type(config.opt3))
    2 1 <class 'int'> <class 'int'>

    >>> parse_config('test.json', **{'opt1__opt2': '2', 'opt3': 1.0}, strict_cast=False)
    >>> print(config.opt1.opt2, config.opt3, type(config.opt1.opt2), type(config.opt3))
    2 1.0 <class 'str'> <class 'float'>
    """
    config = _loader(config_fname, verbose)

    if kwargs:
        _print(f'overwriting configurations: {kwargs}', verbose)

    for arg_key, arg_val in kwargs.items():
        keys = arg_key.split(delimiter)
        n_keys = len(keys)

        _config = config
        for idx, _key in enumerate(keys):
            if n_keys - 1 == idx:
                if strict_cast:
                    typecast = type(_config[_key])
                    _config[_key] = typecast(arg_val)
                else:
                    _config[_key] = arg_val
            else:
                _config = _config[_key]

    config = munch.munchify(config)
    return config
