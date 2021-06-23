"""
PCME
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
from engine import EngineBase


class COCORetrievalEngine(EngineBase):
    def set_gallery_from_dataloader(self, dataloader):
        self.model_to_device()
        extracted_features = self.evaluator.extract_features(dataloader)
        self.image_features = extracted_features['image_features']
        self.caption_features = extracted_features['caption_features']
        self.image_sigmas = extracted_features['image_sigmas']
        self.caption_sigmas = extracted_features['caption_sigmas']
        self.image_ids = extracted_features['image_ids']
        self.caption_ids = extracted_features['caption_ids']
        self.image_classes = extracted_features['image_classes']
        self.caption_classes = extracted_features['caption_classes']

        self.coco = dataloader.dataset.coco

    def retrieve_from_features(self, q_features, q_modality, q_ids=None, q_sigmas=None, topk=10, batch_size=1024):
        if q_modality not in ['image', 'caption']:
            raise ValueError

        if q_modality == 'image':
            g_features = self.caption_features
            g_ids = self.caption_ids
            g_classes = self.caption_classes
            q_classes = self.image_classes

            g_meta_loader = self.coco.loadAnns
            q_meta_loader = self.coco.loadImgs
            q_gt_loader = self.coco.imgToAnns
        elif q_modality == 'caption':
            g_features = self.image_features
            g_ids = self.image_ids
            g_classes = self.image_classes
            q_classes = self.caption_classes

            g_meta_loader = self.coco.loadImgs
            q_meta_loader = self.coco.loadAnns
            q_gt_loader = None

        if q_ids is None:
            q_ids = [idx for idx in range(len(q_features))]

        if q_sigmas is None:
            q_sigmas = [0 for _ in range(len(q_features))]

        retrieved_items, scores, metas = self.evaluator.retrieve(q_features,
                                                                 g_features,
                                                                 q_ids,
                                                                 g_ids,
                                                                 q_classes,
                                                                 g_classes,
                                                                 topk=topk,
                                                                 batch_size=batch_size)

        ret = {}
        for q_sigma, (q_id, items) in zip(q_sigmas, retrieved_items.items()):
            q_id = int(q_id)

            meta = {}
            meta['retrieved'] = g_meta_loader(items)
            meta['scores'] = scores[q_id]
            _meta = q_meta_loader(q_id)[0].copy()
            if q_gt_loader:
                _meta.update({'gt_captions': [item['caption'] for item in q_gt_loader[q_id]]})
            else:
                _meta.update({'gt_image': g_meta_loader(_meta['image_id'])[0]})
            meta['query'] = _meta
            meta['query_sigma'] = q_sigma
            ret[q_id] = meta
        return ret
