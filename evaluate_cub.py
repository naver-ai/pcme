"""
PCME
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import fire
import torch

from logger import PythonLogger

from config import parse_config
from datasets import prepare_cub_dataloaders
from engine import TrainerEngine
from engine import CUBEvaluator


@torch.no_grad()
def evaluate(config, dataset_name, model_path, dataloader, vocab, logger):
    logger.log('start evaluation')
    engine = TrainerEngine()
    engine.set_logger(logger)

    evaluator = CUBEvaluator(eval_method=config.model.get('eval_method', 'matmul'),
                             verbose=True,
                             eval_device='cuda')
    engine.create(config, vocab.word2idx, evaluator)

    engine.load_models(model_path,
                       load_keys=['model', 'criterion'])

    scores = engine.evaluate(val_loaders=dataloader)
    logger.pretty_log_dict(scores)
    return scores


def main(config_path,
         dataset_root,
         caption_root,
         model_path,
         dataset_name='cub',
         split='val',
         vocab_path='datasets/vocabs/cub_vocab.pkl',
         cache_dir='/home/.cache/torch/checkpoints',
         dump_to=None,
         **kwargs):
    config = parse_config(config_path,
                          strict_cast=False,
                          model__cache_dir=cache_dir,
                          **kwargs)

    logger = PythonLogger()
    logger.log('preparing data loaders..')
    dataloaders, vocab = prepare_cub_dataloaders(config.dataloader,
                                                 dataset_name,
                                                 dataset_root,
                                                 caption_root,
                                                 vocab_path)
    scores = evaluate(config, dataset_name, model_path, dataloaders[split], vocab, logger)
    if dump_to:
        torch.save(scores, dump_to)


if __name__ == '__main__':
    fire.Fire(main)
