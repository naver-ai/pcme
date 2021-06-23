"""Cross-modal retrieval evaluation wrapper.

PCME
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.tensor_utils import to_numpy


def batch(iterable, batch_size=1):
    """a batch generator
    """
    n_items = len(iterable)
    for batch_idx in range(0, n_items, batch_size):
        yield iterable[batch_idx:min(batch_idx + batch_size, n_items)]


def recall_at_k(ranks, k):
    """Compute recall at K

    args:
        ranks (list): list of rankings of positive pairs
        k (int): k
    """
    return 100.0 * len(np.where(ranks < k)[0]) / len(ranks)


class ParallelMatMulModule(nn.Module):
    def set_g_features(self, g_features):
        self._g_features = g_features
        self.g_features = None

    def forward(self, q_features, n_embeddings=1):
        if self.g_features is None:
            self.g_features = self._g_features.to(q_features.device)
        sims = q_features.mm(self.g_features)

        if n_embeddings > 1:
            sims = sims.view(int(len(q_features) / n_embeddings),
                             n_embeddings,
                             int(self.g_features.size()[-1] / n_embeddings),
                             n_embeddings)
            sims = sims.permute(0, 1, 3, 2)
            sims = torch.sum(torch.sum(sims, axis=1), axis=1)

        sims, pred_ranks = (-sims).sort()
        return sims, pred_ranks


class MatchingProbModule(nn.Module):
    def __init__(self, match_prob_fn):
        super().__init__()
        self.match_prob_fn = match_prob_fn

    def set_g_features(self, g_features):
        self._g_features = g_features
        self.g_features = None

    def forward(self, q_features, n_embeddings=1):
        if self.g_features is None:
            self.g_features = self._g_features.to(q_features.device)
        sims = torch.zeros(len(q_features), len(self.g_features))
        for idx, q_feature in enumerate(q_features):
            _sim = self.match_prob_fn(q_feature.unsqueeze(0), self.g_features, None, None)
            sims[idx] = _sim
        sims, pred_ranks = (-sims).sort()
        return sims, pred_ranks


class COCOEvaluator(object):
    """COCOEvaluator wrapper

    Args:
        eval_method (str): distance function to use (matmul | matching_prob)
        n_crossfolds (int): default crossfold setting (-1 | 5)
    """
    def __init__(self,
                 eval_method='matmul',
                 n_crossfolds=-1,
                 extract_device='cuda',
                 eval_device='cuda',
                 verbose=False):
        self.eval_method = eval_method
        self.extract_device = extract_device
        self.eval_device = eval_device
        self.logger = None

        self.n_crossfolds = n_crossfolds

        self.pbar = partial(tqdm, disable=not verbose)

    def set_model(self, model):
        """set model
        """
        self.model = model

        if isinstance(self.model, nn.DataParallel):
            self.n_embeddings = self.model.module.n_embeddings
            self.feat_size = self.model.module.embed_dim
        else:
            self.n_embeddings = self.model.n_embeddings
            self.feat_size = self.model.embed_dim

    def set_criterion(self, criterion):
        """set criterion (for matching prob retrieval)
        """
        self.criterion = criterion

    def set_logger(self, logger):
        """set logger
        """
        self.logger = logger

    @torch.no_grad()
    def extract_features(self, dataloader):
        """Extract image and caption features using the given model.

        Args:
            model (nn.Module): a model to extract features.
            dataloader (data.Dataloader): the target dataloader to feature extraction.
        """
        self.model.eval()
        self.model.to(self.extract_device)

        num_images = dataloader.dataset.n_images
        num_captions = len(dataloader.dataset)

        image_classes = np.zeros(num_images)
        caption_classes = np.zeros(num_captions)

        image_features = np.zeros((num_images, self.n_embeddings, self.feat_size))
        caption_features = np.zeros((num_captions, self.n_embeddings, self.feat_size))

        image_sigmas = np.zeros((num_images, self.feat_size))
        caption_sigmas = np.zeros((num_captions, self.feat_size))

        image_ids_ = np.zeros(num_images)
        caption_ids = np.zeros(num_captions)

        cur_image_idx = 0
        cur_caption_idx = 0
        seen_image_ids = set()
        iid_to_cls = dataloader.dataset.iid_to_cls

        def get_image_class(image_id):
            if iid_to_cls:
                image_class = iid_to_cls.get(image_id, image_id)
            else:
                image_class = image_id
            return image_class

        for images, captions, caption_lens, ann_ids, image_ids in self.pbar(dataloader):
            images = images.to(self.extract_device)
            captions = captions.to(self.extract_device)
            caption_lens = caption_lens.to(self.extract_device)

            output = self.model(images, captions, caption_lens)
            _image_features = output['image_features']
            _caption_features = output['caption_features']

            if output.get('image_logsigma') is not None:
                _image_sigmas = output['image_logsigma']
                _caption_sigmas = output['caption_logsigma']

            for idx, image_id in enumerate(image_ids):
                image_class = get_image_class(image_id)
                if image_id not in seen_image_ids:
                    image_ids_[cur_image_idx] = image_id
                    seen_image_ids.add(image_id)
                    image_classes[cur_image_idx] = image_class
                    image_features[cur_image_idx] = to_numpy(_image_features[idx])
                    if output.get('image_logsigma') is not None:
                        image_sigmas[cur_image_idx] = to_numpy(_image_sigmas[idx])
                    cur_image_idx += 1
                caption_ids[cur_caption_idx] = ann_ids[idx]
                caption_classes[cur_caption_idx] = image_class
                caption_features[cur_caption_idx] = to_numpy(_caption_features[idx])
                if output.get('image_logsigma') is not None:
                    caption_sigmas[cur_caption_idx] = to_numpy(_caption_sigmas[idx])
                cur_caption_idx += 1

        if iid_to_cls:
            print(f'Num images ({num_images}) -> Num classes ({len(set(image_classes))})')
        if cur_image_idx != num_images:
            raise RuntimeError('unexpected error, {} != {}'.format(cur_image_idx, num_images))
        if cur_caption_idx != num_captions:
            raise RuntimeError('unexpected error, {}, {}'.format(cur_caption_idx, num_captions))
        if set(image_classes) != set(caption_classes):
            raise RuntimeError('unexpected error, I({}) != C({})'.format(set(image_classes), set(caption_classes)))

        if not iid_to_cls:
            # XXX this code is for aligning image features and caption features
            # but if you use classes as COCO classes, but image_id,
            # the sorted_caption_idx will return multiple instances, and
            # the results will be corrupted.
            sorted_caption_idx = []
            for image_class in image_classes:
                sorted_caption_idx.extend(np.where(caption_classes == image_class)[0])

            sorted_caption_idx = np.array(sorted_caption_idx)
            caption_ids = caption_ids[sorted_caption_idx]
            caption_classes = caption_classes[sorted_caption_idx]
            caption_features = caption_features[sorted_caption_idx]

        image_features = torch.from_numpy(image_features)
        caption_features = torch.from_numpy(caption_features)
        image_classes = torch.from_numpy(image_classes)
        caption_classes = torch.from_numpy(caption_classes)

        return {
            'image_features': image_features,
            'caption_features': caption_features,
            'image_sigmas': image_sigmas,
            'caption_sigmas': caption_sigmas,
            'image_ids': image_ids_,
            'caption_ids': caption_ids,
            'image_classes': image_classes,
            'caption_classes': caption_classes,
        }

    @torch.no_grad()
    def retrieve(self, q_features, g_features,
                 q_ids, g_ids,
                 q_classes=None, g_classes=None,
                 topk=10,
                 batch_size=1024):
        if len(q_features) != len(q_ids):
            raise RuntimeError('length mismatch {}, {}'.format(q_features.shape,
                                                               q_ids.shape))
        if len(g_features) != len(g_ids):
            raise RuntimeError('length mismatch {}, {}'.format(g_features.shape,
                                                               g_ids.shape))
        if isinstance(q_ids, list) or isinstance(g_ids, list):
            q_ids = np.array(q_ids)
            g_ids = np.array(g_ids)
        n_queries = len(q_ids)
        n_galleries = len(g_ids)

        if self.eval_method == 'matmul':
            pmm = ParallelMatMulModule()
            g_features = g_features.view(n_galleries * self.n_embeddings, -1).t()
        elif self.eval_method == 'matching_prob':
            pmm = MatchingProbModule(self.criterion.match_prob)
        pmm.set_g_features(g_features)

        q_features = q_features.to(self.eval_device)

        retrieved_items = {}
        retrieved_scores = {}

        for q_indices in self.pbar(batch(range(n_queries), batch_size=batch_size)):
            q_indices = np.array(q_indices)

            if self.eval_method != 'matching_prob':
                _q_feature = q_features[q_indices, :]
                _q_feature = _q_feature.view(len(q_indices) * self.n_embeddings, -1)
            else:
                _q_feature = q_features[q_indices, :, :]
            sims, pred_ranks = pmm(_q_feature, n_embeddings=self.n_embeddings)

            _, sorted_indices = pred_ranks.sort()
            for idx, sorted_db_index in enumerate(sorted_indices):
                _, _sorted_idx = sorted_db_index.sort()
                retrieved_items[q_ids[q_indices[idx]]] = [item for item in g_ids[to_numpy(_sorted_idx)[0][:topk]]]
                retrieved_scores[q_ids[q_indices[idx]]] = sims[idx][:topk].cpu().numpy()

        return retrieved_items, retrieved_scores, None

    @torch.no_grad()
    def evaluate_recall(self, q_features, g_features, q_labels, g_labels,
                        q_ids=None, g_ids=None,
                        batch_size=1024):
        """Evaluate recall

        Args:
            q_features (tensor): N_q x d query features
            g_features (tensor): N_g x d gallery features
            q_labels (tensor): N query labels
            g_labels (tensor): N gallery labels
        """
        if len(q_features) != len(q_labels):
            raise RuntimeError('length mismatch {}, {}'.format(q_features.shape,
                                                               q_labels.shape))
        if len(g_features) != len(g_labels):
            raise RuntimeError('length mismatch {}, {}'.format(g_features.shape,
                                                               g_labels.shape))
        n_queries = len(q_labels)
        n_galleries = len(g_labels)
        best_pred_ranks = np.zeros(n_queries)

        if self.eval_method == 'matmul':
            pmm = ParallelMatMulModule()
            g_features = g_features.view(n_galleries * self.n_embeddings, -1).t()
        elif self.eval_method == 'matching_prob':
            pmm = MatchingProbModule(self.criterion.match_prob)
        pmm.set_g_features(g_features)

        q_features = q_features.to(self.eval_device)

        for q_indices in self.pbar(batch(range(n_queries), batch_size=batch_size)):
            q_indices = np.array(q_indices)

            if self.eval_method != 'matching_prob':
                _q_feature = q_features[q_indices, :]
                _q_feature = _q_feature.view(len(q_indices) * self.n_embeddings, -1)
            else:
                _q_feature = q_features[q_indices, :, :]
            _, pred_ranks = pmm(_q_feature, n_embeddings=self.n_embeddings)

            for idx, q_idx in enumerate(q_indices):
                pos_indices = np.where(g_labels == q_labels[q_idx])[0]
                _pred_ranks = [torch.where(pred_ranks[idx] == pos_idx)[0][0].item() for pos_idx in pos_indices]
                best_pred_ranks[q_idx] = min(_pred_ranks)

        recall_1 = recall_at_k(best_pred_ranks, 1)
        recall_5 = recall_at_k(best_pred_ranks, 5)
        recall_10 = recall_at_k(best_pred_ranks, 10)
        medr = np.floor(np.median(best_pred_ranks)) + 1
        meanr = np.mean(best_pred_ranks) + 1

        scores = {
            'recall_1': recall_1,
            'recall_5': recall_5,
            'recall_10': recall_10,
            'rsum': recall_1 + recall_5 + recall_10,
            'medr': medr,
            'meanr': meanr,
        }

        return scores

    def evaluate_n_fold(self, extracted_features, n_crossfolds, n_images_per_crossfold,
                        n_captions_per_crossfold, eval_batch_size):
        image_features = extracted_features['image_features']
        caption_features = extracted_features['caption_features']
        image_classes = extracted_features['image_classes']
        caption_classes = extracted_features['caption_classes']

        n_fold_scores = {
            'i2t': {
                'recall_1': [],
                'recall_5': [],
                'recall_10': [],
                'rsum': [],
                'medr': [],
                'meanr': [],
            },
            't2i': {
                'recall_1': [],
                'recall_5': [],
                'recall_10': [],
                'rsum': [],
                'medr': [],
                'meanr': [],
            },
        }

        for idx in range(n_crossfolds):
            if self.logger:
                self.logger.log('evaluating {}-th fold'.format(idx + 1))

            _image_split = np.arange(idx * n_images_per_crossfold, (idx + 1) * n_images_per_crossfold)
            _image_features = image_features[_image_split]
            _image_classes = image_classes[_image_split]

            _caption_split = np.arange(idx * n_captions_per_crossfold, (idx + 1) * n_captions_per_crossfold)
            _caption_features = caption_features[_caption_split]
            _caption_classes = caption_classes[_caption_split]

            _scores = {}
            _scores['i2t'] = self.evaluate_recall(_image_features,
                                                  _caption_features,
                                                  _image_classes,
                                                  _caption_classes,
                                                  batch_size=eval_batch_size)
            _scores['t2i'] = self.evaluate_recall(_caption_features,
                                                  _image_features,
                                                  _caption_classes,
                                                  _image_classes,
                                                  batch_size=eval_batch_size)
            for _task, _task_scores in _scores.items():
                for key, val in _task_scores.items():
                    n_fold_scores[_task][key].append(val)
        n_fold_scores = {_task: {key: np.mean(np.array(val)) for key, val in _task_scores.items()}
                         for _task, _task_scores in n_fold_scores.items()}
        return n_fold_scores

    @torch.no_grad()
    def evaluate(self, dataloader, n_crossfolds=None,
                 n_images_per_crossfold=1000,
                 n_captions_per_crossfold=5000,
                 eval_batch_size=1024,
                 key=None):
        """evaluate image-to-caption and caption-to-image retrieval tasks.
        """
        scores = {}

        if self.logger:
            self.logger.log('extracting features...')

        extracted_features = self.extract_features(dataloader)

        image_features = extracted_features['image_features']
        caption_features = extracted_features['caption_features']
        image_sigmas = extracted_features['image_sigmas']
        caption_sigmas = extracted_features['caption_sigmas']
        image_classes = extracted_features['image_classes']
        caption_classes = extracted_features['caption_classes']

        scores['mean_log_image_sigma'] = np.mean(image_sigmas)
        scores['mean_log_caption_sigma'] = np.mean(caption_sigmas)

        if n_crossfolds is None:
            n_crossfolds = self.n_crossfolds

        if dataloader.dataset.iid_to_cls:
            print('"use_class" setting does not evaluate 1k crossfolds')
            n_crossfolds = -1

        if n_crossfolds > 0:
            n_fold_scores = self.evaluate_n_fold(extracted_features,
                                                 n_crossfolds,
                                                 n_images_per_crossfold,
                                                 n_captions_per_crossfold,
                                                 eval_batch_size)
            scores['n_fold'] = n_fold_scores

        if self.logger:
            self.logger.log('evaluating i2t...')
        scores['i2t'] = self.evaluate_recall(image_features,
                                             caption_features,
                                             image_classes,
                                             caption_classes,
                                             batch_size=eval_batch_size)
        if self.logger:
            self.logger.log('evaluating t2i...')
        scores['t2i'] = self.evaluate_recall(caption_features,
                                             image_features,
                                             caption_classes,
                                             image_classes,
                                             batch_size=eval_batch_size)
        for key in ('rsum', 'medr', 'meanr'):
            scores[key] = scores['i2t'][key] + scores['t2i'][key]
        return scores
