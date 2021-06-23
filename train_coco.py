"""End-to-end training code for cross-modal retrieval tasks.

PCME
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import os
import fire

import torch.backends.cudnn as cudnn

from config import parse_config
from datasets import prepare_coco_dataloaders
from engine import TrainerEngine
from engine import COCOEvaluator
from logger import PythonLogger


def pretrain(config, dataloaders, vocab, logger):
    logger.log('start pretrain')
    engine = TrainerEngine()
    engine.set_logger(logger)

    config.model.img_finetune = False
    config.model.txt_finetune = False

    _dataloaders = dataloaders.copy()

    val_epochs = config.train.get('pretrain_val_epochs', 1)
    evaluator = COCOEvaluator(eval_method=config.model.get('eval_method', 'matmul'),
                              verbose=False,
                              eval_device='cuda',
                              n_crossfolds=5)
    engine.create(config, vocab.word2idx, evaluator)

    engine.train(tr_loader=_dataloaders.pop('train'),
                 n_epochs=config.train.pretrain_epochs,
                 val_loaders=_dataloaders,
                 val_epochs=val_epochs,
                 model_save_to=config.train.pretrain_save_path,
                 best_model_save_to=config.train.best_pretrain_save_path)


def finetune(config, pretrain_path, dataloaders, vocab, logger):
    logger.log('start finetune')
    engine = TrainerEngine()
    engine.set_logger(logger)

    config.model.img_finetune = True
    config.model.txt_finetune = True
    config.optimizer.learning_rate *= config.train.get('finetune_lr_decay', 0.1)

    _dataloaders = dataloaders.copy()

    val_epochs = config.train.get('val_epochs', 1)

    evaluator = COCOEvaluator(eval_method=config.model.get('eval_method', 'matmul'),
                              verbose=False,
                              eval_device='cuda',
                              n_crossfolds=5)
    engine.create(config, vocab.word2idx, evaluator)

    if os.path.exists(pretrain_path):
        engine.load_models(pretrain_path,
                           load_keys=['model', 'criterion'])

    engine.train(tr_loader=_dataloaders.pop('train'),
                 n_epochs=config.train.finetune_epochs,
                 val_loaders=_dataloaders,
                 val_epochs=val_epochs,
                 model_save_to=config.train.model_save_path,
                 best_model_save_to=config.train.best_model_save_path)


def main(config_path,
         dataset_root,
         vocab_path='./datasets/vocabs/coco_vocab.pkl',
         **kwargs):
    """Main interface for the training.

    Args:
        config_path: path to the configuration file
        dataset_root: root for the dataset
        vocab_path: vocab filename

        Other configurations:
            you can override any pcme configuration in the command line!
            try, --<depth1>__<depth2>. E.g., --dataloader__batch_size 32
    """
    logger = PythonLogger()

    config = parse_config(config_path,
                          strict_cast=False,
                          **kwargs)
    cudnn.benchmark = True

    dataloaders, vocab = prepare_coco_dataloaders(config.dataloader,
                                                  dataset_root, vocab_path)

    pretrain(config, dataloaders, vocab, logger)
    finetune(config, config.train.pretrain_save_path,
             dataloaders, vocab, logger)


if __name__ == '__main__':
    fire.Fire(main)
