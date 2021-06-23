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
from engine import COCOEvaluator
from engine import COCORetrievalEngine


@torch.no_grad()
def main(config_path,
         dataset_root,
         model_path,
         dump_to,
         vocab_path='datasets/vocabs/coco_vocab.pkl',
         cache_dir='/home/.cache/torch/checkpoints',
         dump_features_to=None,
         split='te',
         topk=-1,
         **kwargs):
    dt = datetime.datetime.now()

    config = parse_config(config_path,
                          strict_cast=False,
                          model__cache_dir=cache_dir,
                          **kwargs)

    logger = PythonLogger()
    logger.log('preparing data loaders..')
    dataloaders, vocab = prepare_coco_dataloaders(config.dataloader,
                                                  dataset_root, vocab_path)

    engine = COCORetrievalEngine()
    engine.set_logger(logger)

    evaluator = COCOEvaluator(eval_method='matching_prob',
                              verbose=True,
                              eval_device='cuda',
                              n_crossfolds=5)
    engine.create(config, vocab.word2idx, evaluator)
    engine.load_models(model_path,
                       load_keys=['model', 'criterion'])
    engine.set_gallery_from_dataloader(dataloaders[split])

    if dump_features_to:
        torch.save({
            'images': engine.image_features,
            'captions': engine.caption_features,
            'image_ids': engine.image_ids,
            'image_sigmas': engine.image_sigmas,
            'image_classes': engine.image_classes,
            'caption_ids': engine.caption_ids,
            'caption_sigmas': engine.caption_sigmas,
            'caption_classes': engine.caption_classes,
        }, dump_features_to)

    i2t_retrieved_items = engine.retrieve_from_features(engine.image_features,
                                                        q_modality='image',
                                                        q_ids=engine.image_ids,
                                                        q_sigmas=engine.image_sigmas,
                                                        topk=topk,
                                                        batch_size=config.dataloader.eval_batch_size,
                                                        )
    t2i_retrieved_items = engine.retrieve_from_features(engine.caption_features,
                                                        q_modality='caption',
                                                        q_ids=engine.caption_ids,
                                                        q_sigmas=engine.caption_sigmas,
                                                        topk=topk,
                                                        batch_size=config.dataloader.eval_batch_size,
                                                        )

    data = {
        'i2t': i2t_retrieved_items,
        't2i': t2i_retrieved_items,
    }

    torch.save(data, dump_to)
    logger.log('takes {}'.format(datetime.datetime.now() - dt))


if __name__ == '__main__':
    fire.Fire(main)
