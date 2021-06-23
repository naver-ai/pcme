"""
PCME
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import datetime

import fire
import torch

from logger import PythonLogger

from config import parse_config
from datasets import prepare_coco_dataloaders
from engine import TrainerEngine
from engine import COCOEvaluator


@torch.no_grad()
def evaluate(config, model_path, dataloader, vocab, logger, n_crossfolds):
    logger.log('start evaluation')
    engine = TrainerEngine()
    engine.set_logger(logger)

    config.model.img_finetune = False
    config.model.txt_finetune = False

    evaluator = COCOEvaluator(eval_method='matching_prob',
                              verbose=True,
                              eval_device='cuda',
                              n_crossfolds=n_crossfolds)
    engine.create(config, vocab.word2idx, evaluator)
    engine.load_models(model_path,
                       load_keys=['model', 'criterion'])

    scores = engine.evaluate(val_loaders=dataloader,
                             eval_batch_size=config.dataloader.eval_batch_size)
    logger.pretty_log_dict(scores)
    return scores


def main(config_path,
         dataset_root,
         model_path,
         n_crossfolds,
         split='te',
         vocab_path='datasets/vocabs/coco_vocab.pkl',
         cache_dir='/home/.cache/torch/checkpoints',
         dump_to=None,
         **kwargs):
    if n_crossfolds not in {-1, 5}:
        raise ValueError(f'n_crossfolds should be in (-1, 5) not {n_crossfolds}')
    dt = datetime.datetime.now()
    config = parse_config(config_path,
                          strict_cast=False,
                          model__cache_dir=cache_dir,
                          **kwargs)

    logger = PythonLogger()
    logger.log('preparing data loaders..')
    dataloaders, vocab = prepare_coco_dataloaders(config.dataloader,
                                                  dataset_root, vocab_path)
    dataloader = dataloaders[split]
    scores = evaluate(config, model_path, dataloader, vocab, logger, n_crossfolds)
    if dump_to:
        torch.save(scores, dump_to)

    logger.log('takes {}'.format(datetime.datetime.now() - dt))


if __name__ == '__main__':
    fire.Fire(main)
