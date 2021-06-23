"""
PCME
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import hashlib
import json
import munch

import torch

from config import parse_config

from criterions import get_criterion
from models import get_model
from optimizers import get_optimizer
from optimizers import get_lr_scheduler

from utils.serialize_utils import torch_safe_load

try:
    from apex import amp
except ImportError:
    print('failed to import apex')


class EngineBase(object):
    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.lr_scheduler = None
        self.evaluator = None

        self.config = None
        self.logger = None

        self.metadata = {}

    def create(self, config, word2idx, evaluator, verbose=False):
        self.config = config
        self.word2idx = word2idx
        self.set_model(get_model(config.model.name,
                                 word2idx,
                                 config.model))
        self.set_criterion(get_criterion(config.criterion.name,
                                         config.criterion))
        params = [param for param in self.model.parameters()
                  if param.requires_grad]
        params += [param for param in self.criterion.parameters()
                   if param.requires_grad]
        self.set_optimizer(get_optimizer(config.optimizer.name,
                                         params,
                                         config.optimizer))
        self.set_lr_scheduler(get_lr_scheduler(config.lr_scheduler.name,
                                               self.optimizer,
                                               config.lr_scheduler))
        evaluator.set_model(self.model)
        evaluator.set_criterion(self.criterion)
        self.set_evaluator(evaluator)

        self.logger.log('Engine is created.')
        self.logger.log(config)

        self.logger.update_tracker({'full_config': munch.unmunchify(config)}, keys=['full_config'])

    def set_model(self, model):
        self.model = model

    def model_to_device(self):
        self.model.to(self.device)
        if self.criterion:
            self.criterion.to(self.device)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_criterion(self, criterion):
        self.criterion = criterion

    def set_lr_scheduler(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    def set_evaluator(self, evaluator):
        self.evaluator = evaluator
        self.evaluator.set_logger(self.logger)

    def set_logger(self, logger):
        self.logger = logger

    def to_half(self):
        # Mixed precision
        # https://nvidia.github.io/apex/amp.html
        self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                                                    opt_level='O2')

    @torch.no_grad()
    def evaluate(self, val_loaders, n_crossfolds=None, **kwargs):
        if self.evaluator is None:
            self.logger.log('[Evaluate] Warning, no evaluator is defined. Skip evaluation')
            return

        self.model_to_device()
        self.model.eval()

        if not isinstance(val_loaders, dict):
            val_loaders = {'te': val_loaders}

        scores = {}
        for key, data_loader in val_loaders.items():
            self.logger.log('Evaluating {}...'.format(key))
            _n_crossfolds = -1 if key == 'val' else n_crossfolds
            scores[key] = self.evaluator.evaluate(data_loader, n_crossfolds=_n_crossfolds,
                                                  key=key, **kwargs)
        return scores

    def save_models(self, save_to, metadata=None):
        state_dict = {
            'model': self.model.state_dict(),
            'criterion': self.criterion.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'config': munch.unmunchify(self.config),
            'word2idx': self.word2idx,
            'metadata': metadata,
        }
        torch.save(state_dict, save_to)
        self.logger.log('state dict is saved to {}, metadata: {}'.format(
            save_to, json.dumps(metadata, indent=4)))

    def load_models(self, state_dict_path, load_keys=None):
        with open(state_dict_path, 'rb') as fin:
            model_hash = hashlib.sha1(fin.read()).hexdigest()
            self.metadata['pretrain_hash'] = model_hash

        state_dict = torch.load(state_dict_path, map_location='cpu')

        if 'model' not in state_dict:
            torch_safe_load(self.model, state_dict, strict=False)
            return

        if not load_keys:
            load_keys = ['model', 'criterion', 'optimizer', 'lr_scheduler']
        for key in load_keys:
            try:
                torch_safe_load(getattr(self, key), state_dict[key])
            except RuntimeError as e:
                self.logger.log('Unable to import state_dict, missing keys are found. {}'.format(e))
                torch_safe_load(getattr(self, key), state_dict[key], strict=False)
        self.logger.log('state dict is loaded from {} (hash: {}), load_key ({})'.format(state_dict_path,
                                                                                        model_hash,
                                                                                        load_keys))

    def load_state_dict(self, state_dict_path, load_keys=None):
        state_dict = torch.load(state_dict_path)
        config = parse_config(state_dict['config'])
        self.create(config, state_dict['word2idx'])
        self.load_models(state_dict_path, load_keys)
