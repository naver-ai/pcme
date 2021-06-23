"""
PCME
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import datetime

import torch.nn as nn

from engine import EngineBase
from utils.serialize_utils import flatten_dict

try:
    from apex import amp
except ImportError:
    print('failed to import apex')


def cur_step(cur_epoch, idx, N, fmt=None):
    _cur_step = cur_epoch + idx / N
    if fmt:
        return fmt.format(_cur_step)
    else:
        return _cur_step


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class TrainerEngine(EngineBase):
    def _train_epoch(self, dataloader, cur_epoch, prefix=''):
        self.model.train()
        for idx, (images, captions, caption_lens, _, _) in enumerate(dataloader):
            images = images.to(self.device)
            captions = captions.to(self.device)
            caption_lens = caption_lens.to(self.device)

            output = self.model(images, captions, caption_lens)
            loss, loss_dict = self.criterion(**output)

            self.optimizer.zero_grad()

            if self.config.train.get('use_fp16'):
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if self.config.train.grad_clip > 0:
                nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(),
                                                   self.config.train.grad_clip)
            self.optimizer.step()

            if (idx + 1) % self.config.train.log_step == 0:
                loss_dict = {'{}{}'.format(prefix, key): val
                             for key, val in loss_dict.items()}
                loss_dict['step'] = cur_step(cur_epoch, idx, len(dataloader))
                self.logger.report(loss_dict,
                                   prefix='[Train] Report @step: ')

    def train(self, tr_loader, n_epochs,
              val_loaders=None,
              val_epochs=1,
              model_save_to='last.pth',
              best_model_save_to='best.pth'):

        if val_loaders and 'val' not in val_loaders:
            raise KeyError('val_loaders should contain key "val", '
                           'but ({})'.format(val_loaders.keys()))

        dt = datetime.datetime.now()

        if self.config.model.img_finetune:
            prefix = 'train__'
            eval_prefix = ''
            self.logger.log('start train')
        else:
            prefix = 'pretrain__'
            eval_prefix = 'pretrain_'
            self.logger.log('start pretrain')

        self.model_to_device()
        if self.config.train.get('use_fp16'):
            self.logger.log('Train with half precision')
            self.to_half()

        best_score = 0
        for cur_epoch in range(n_epochs):
            self._train_epoch(tr_loader, cur_epoch, prefix=prefix)

            metadata = self.metadata.copy()
            metadata['cur_epoch'] = cur_epoch + 1
            metadata['lr'] = get_lr(self.optimizer)

            if val_loaders is not None and (cur_epoch + 1) % val_epochs == 0:
                scores = self.evaluate(val_loaders)
                metadata['scores'] = scores['val']

                if best_score < scores['val']['rsum']:
                    self.save_models(best_model_save_to, metadata)
                    best_score = scores['val']['rsum']
                    metadata['best_score'] = best_score
                    metadata['best_epoch'] = cur_epoch + 1

                self.report_scores(step=cur_epoch + 1,
                                   scores=scores,
                                   metadata=metadata,
                                   prefix=eval_prefix)

            if self.config.lr_scheduler.name == 'reduce_lr_on_plateau':
                self.lr_scheduler.step(scores['val']['rsum'])
            else:
                self.lr_scheduler.step()

            self.save_models(model_save_to, metadata)

            elasped = datetime.datetime.now() - dt
            expected_total = elasped / (cur_epoch + 1) * n_epochs
            expected_remain = expected_total - elasped
            self.logger.log('expected remain {}'.format(expected_remain))
        self.logger.log('finish train, takes {}'.format(datetime.datetime.now() - dt))

    def report_scores(self, step, scores, metadata, prefix=''):
        report_dict = {data_key: flatten_dict(_scores, sep='_')
                       for data_key, _scores in scores.items()}
        report_dict = flatten_dict(report_dict, sep='__')
        tracker_data = report_dict.copy()

        report_dict = {'{}{}'.format(prefix, key): val for key, val in report_dict.items()}
        report_dict['step'] = step
        if 'lr' in metadata:
            report_dict['{}lr'.format(prefix)] = metadata['lr']

        self.logger.report(report_dict,
                           prefix='[Eval] Report @step: ',
                           pretty=True)

        tracker_data['metadata'] = metadata
        tracker_data['scores'] = scores
        self.logger.update_tracker(tracker_data)
