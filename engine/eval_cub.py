"""Cross-modal retrieval evaluation wrapper.

reference code:

- https://github.com/KevinMusgrave/pytorch-metric-learning/blob/0b575b556fe339c2a62043d0ff0efe7fe85107bc/src/pytorch_metric_learning/utils/accuracy_calculator.py#L45
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


def recall_at_k(knn_labels, gt_labels, k):
    curr_knn_labels = knn_labels[:, :k]
    accuracy_per_sample = np.any(curr_knn_labels == gt_labels, axis=1)

    return np.mean(accuracy_per_sample)


def precision_at_k(knn_labels, gt_labels, k):
    curr_knn_labels = knn_labels[:, :k]
    accuracy_per_sample = np.sum(curr_knn_labels == gt_labels, axis=1) / k
    return np.mean(accuracy_per_sample)


def get_relevance_mask(shape, gt_labels, label_counts, embeddings_come_from_same_source):
    relevance_mask = np.zeros(shape=shape, dtype=np.int)
    for k, max_column in label_counts.items():
        matching_rows = np.where(gt_labels == k)[0]
        if embeddings_come_from_same_source:
            max_column = max_column - 1
        relevance_mask[matching_rows, :max_column] = 1
    return relevance_mask


def r_precision(knn_labels, gt_labels, embeddings_come_from_same_source, label_counts):
    relevance_mask = get_relevance_mask(
        knn_labels.shape, gt_labels, label_counts, embeddings_come_from_same_source
    )
    matches_per_row = np.sum(
        (knn_labels == gt_labels) * relevance_mask.astype(bool), axis=1
    )
    max_possible_matches_per_row = np.sum(relevance_mask, axis=1)
    accuracy_per_sample = matches_per_row / max_possible_matches_per_row
    return accuracy_per_sample


class ParallelMatMulModule(nn.Module):
    def set_g_features(self, g_features, g_sigmas=None):
        self._g_features = g_features
        self.g_features = None

    def forward(self, q_features, n_embeddings=1, reduction=None,
                embeddings_come_from_same_source=False,
                q_indices=None, q_sigmas=None):
        if self.g_features is None:
            self.g_features = self._g_features.to(q_features.device)
        sims = q_features.mm(self.g_features)

        if n_embeddings > 1:
            sims = sims.view(int(len(q_features) / n_embeddings),
                             n_embeddings,
                             int(self.g_features.size()[-1] / n_embeddings),
                             n_embeddings)
            sims = sims.permute(0, 1, 3, 2)

            if reduction == 'sum':
                sims = torch.sum(torch.sum(sims, axis=1), axis=1)
            elif reduction == 'max':
                sims = torch.max(torch.max(sims, axis=1)[0], axis=1)[0]

        if embeddings_come_from_same_source:
            # ignore self
            for idx, qidx in enumerate(q_indices):
                sims[idx, qidx] = -np.inf
        sims, pred_ranks = (-sims).sort()
        return sims, pred_ranks


class MatchingProbModule(nn.Module):
    def __init__(self, match_prob_fn):
        super().__init__()
        self.match_prob_fn = match_prob_fn

    def set_g_features(self, g_features, g_sigmas=None):
        self._g_features = g_features
        self.g_features = None

    def forward(self, q_features, n_embeddings=1, reduction=None,
                embeddings_come_from_same_source=False,
                q_indices=None, q_sigmas=None):
        if self.g_features is None:
            self.g_features = self._g_features.to(q_features.device)
        sims = torch.zeros(len(q_features), len(self.g_features))
        for idx, q_feature in enumerate(q_features):
            _sim = self.match_prob_fn(q_feature.unsqueeze(0), self.g_features, None, None)
            sims[idx] = _sim
        sims, pred_ranks = (-sims).sort()
        return sims, pred_ranks


def elk_dist(mu1, mu2, log_sigma1, log_sigma2):
    sum_sigma_sq = torch.exp(2 * log_sigma1) + torch.exp(2 * log_sigma2)
    dist = (mu1 - mu2) ** 2
    dist = dist.squeeze(1)
    elk = dist / (sum_sigma_sq) + torch.log(sum_sigma_sq)
    return -0.5 * torch.sum(elk, dim=1)


def kl_dist(mu1, mu2, log_sigma1, log_sigma2):
    dist = (mu1 - mu2) ** 2
    dist = dist.squeeze(1)

    var2 = torch.exp(log_sigma2 * 2)
    dist = dist / var2 + 2 * (log_sigma1 - log_sigma2) + torch.exp(log_sigma1 * 2) / var2
    return -torch.sum(dist, dim=1)


def reverse_kl_dist(mu1, mu2, log_sigma1, log_sigma2):
    return kl_dist(mu2, mu1, log_sigma2, log_sigma1)


def js_dist(mu1, mu2, log_sigma1, log_sigma2):
    dist = (mu1 - mu2) ** 2
    dist = dist.squeeze(1)

    var1 = torch.exp(log_sigma1 * 2)
    var2 = torch.exp(log_sigma2 * 2)
    dist = (dist + var1) / var2 + (dist + var2) / var1
    return -torch.sum(dist, dim=1)


def bhattacharyya_dist(mu1, mu2, log_sigma1, log_sigma2):
    dist = (mu1 - mu2) ** 2
    dist = dist.squeeze(1)
    sigma1 = torch.exp(log_sigma1)
    sigma2 = torch.exp(log_sigma2)

    dist = dist / (torch.exp(log_sigma1 * 2) + torch.exp(log_sigma2 * 2))
    dist = dist + 2 * torch.log(sigma1 / sigma2 + sigma2 / sigma1)
    ddd = 2 * torch.log(torch.ones(1) * 2)
    dist = dist.float() - ddd.to(dist.device).float()
    dist = dist / 4
    return -torch.sum(dist, dim=1)


def wasserstein_dist(mu1, mu2, log_sigma1, log_sigma2):
    dist = (mu1 - mu2) ** 2
    dist = dist.squeeze(1)
    dist = dist + (torch.exp(log_sigma1) - torch.exp(log_sigma2)) ** 2
    return -torch.sum(dist, dim=1)


def l2_dist(mu1, mu2, log_sigma1, log_sigma2):
    dist = (mu1 - mu2) ** 2
    dist = dist.squeeze(1)
    return -torch.sum(dist, dim=1)


def min_dist(mu1, mu2, log_sigma1, log_sigma2):
    dist = (mu1 - mu2) ** 2
    dist = dist.squeeze(1)
    return -torch.min(dist, dim=1)[0]


def max_dist(mu1, mu2, log_sigma1, log_sigma2):
    dist = (mu1 - mu2) ** 2
    dist = dist.squeeze(1)
    return -torch.max(dist, dim=1)[0]


class HybridDistanceModule(nn.Module):
    def __init__(self, dist_fn_name):
        super().__init__()
        print(dist_fn_name)
        if dist_fn_name == 'elk':
            self.dist_fn = elk_dist
        elif dist_fn_name == 'l2':
            self.dist_fn = l2_dist
        elif dist_fn_name == 'min':
            self.dist_fn = min_dist
        elif dist_fn_name == 'max':
            self.dist_fn = max_dist
        elif dist_fn_name == 'wasserstein':
            self.dist_fn = wasserstein_dist
        elif dist_fn_name == 'kl':
            self.dist_fn = kl_dist
        elif dist_fn_name == 'reverse_kl':
            self.dist_fn = reverse_kl_dist
        elif dist_fn_name == 'js':
            self.dist_fn = js_dist
        elif dist_fn_name == 'bhattacharyya':
            self.dist_fn = bhattacharyya_dist
        else:
            raise ValueError(dist_fn_name)

    def set_g_features(self, g_features, g_sigmas=None):
        self._g_features = g_features
        self.g_features = None
        self._g_sigmas = torch.from_numpy(g_sigmas)
        self.g_sigmas = None

    def forward(self, q_features, n_embeddings=1, reduction=None,
                embeddings_come_from_same_source=False,
                q_indices=None, q_sigmas=None):
        if self.g_features is None:
            self.g_features = self._g_features.to(q_features.device)
            self.g_sigmas = self._g_sigmas.to(q_features.device)

        sims = torch.zeros(len(q_features), len(self.g_features))

        q_sigmas = torch.from_numpy(q_sigmas).to(q_features.device)
        for idx, (q_feature, q_sigma) in enumerate(zip(q_features, q_sigmas)):
            _sim = self.dist_fn(q_feature.unsqueeze(0), self.g_features, q_sigma.unsqueeze(0), self.g_sigmas)
            sims[idx] = _sim

        sims, pred_ranks = (-sims).sort()
        return sims, pred_ranks


class CUBEvaluator(object):
    """Evaluator wrapper

    Args:
        eval_method (str): distance function to use, should be in
            ('elk', 'l2', 'min', 'max', 'wasserstein', 'kl',
            'reverse_kl', 'js', 'bhattacharyya', 'matmul', 'matching_prob')
        n_crossfolds (int): default crossfold setting (-1 | 5)
    """
    def __init__(self,
                 eval_method='matmul',
                 extract_device='cuda',
                 eval_device='cuda',
                 verbose=False):
        self.eval_method = eval_method
        self.extract_device = extract_device
        self.eval_device = eval_device
        self.logger = None

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
        num_classes = len(dataloader.dataset.class_to_indices)

        image_classes = np.zeros(num_images)
        caption_classes = np.zeros(num_captions)

        image_features = np.zeros((num_images, self.n_embeddings, self.feat_size))
        caption_features = np.zeros((num_captions, self.n_embeddings, self.feat_size))

        image_sigmas = np.zeros((num_images, self.feat_size))
        caption_sigmas = np.zeros((num_captions, self.feat_size))

        print(f'num_images = {num_images}, num_captions = {num_captions}, num_classes = {num_classes}')

        image_paths = []
        raw_captions = []

        cur_image_idx = 0
        cur_caption_idx = 0
        seen_image_ids = set()

        dataset_meta = dataloader.dataset.targets

        for images, captions, caption_lens, class_ids, annotation_ids in self.pbar(dataloader):
            images = images.to(self.extract_device)
            captions = captions.to(self.extract_device)
            caption_lens = caption_lens.to(self.extract_device)

            output = self.model(images, captions, caption_lens)
            _image_features = output['image_features']
            _caption_features = output['caption_features']

            if output.get('image_logsigma') is not None:
                _image_sigmas = output['image_logsigma']
                _caption_sigmas = output['caption_logsigma']

            for idx, (class_id, ann_id) in enumerate(zip(class_ids, annotation_ids)):
                image_id, raw_caption = dataset_meta[ann_id]
                if image_id not in seen_image_ids:
                    image_classes[cur_image_idx] = class_id
                    image_features[cur_image_idx] = to_numpy(_image_features[idx])
                    if output.get('image_logsigma') is not None:
                        image_sigmas[cur_image_idx] = to_numpy(_image_sigmas[idx])
                    cur_image_idx += 1
                    seen_image_ids.add(image_id)
                    image_paths.append(image_id)

                raw_captions.append(raw_caption)
                caption_classes[cur_caption_idx] = class_id
                caption_features[cur_caption_idx] = to_numpy(_caption_features[idx])
                if output.get('image_logsigma') is not None:
                    caption_sigmas[cur_caption_idx] = to_numpy(_caption_sigmas[idx])
                cur_caption_idx += 1

        if cur_image_idx != num_images:
            raise RuntimeError('unexpected error, {} != {}'.format(cur_image_idx, num_images))
        if cur_caption_idx != num_captions:
            raise RuntimeError('unexpected error, {}, {}'.format(cur_caption_idx, num_captions))
        if set(image_classes) != set(caption_classes):
            raise RuntimeError('unexpected error, I({}) != C({})'.format(set(image_classes), set(caption_classes)))

        image_features = torch.from_numpy(image_features)
        caption_features = torch.from_numpy(caption_features)
        image_classes = torch.from_numpy(image_classes)
        caption_classes = torch.from_numpy(caption_classes)

        return {
            'image_features': image_features,
            'caption_features': caption_features,
            'image_sigmas': image_sigmas,
            'caption_sigmas': caption_sigmas,
            'image_ids': image_paths,
            'caption_ids': raw_captions,
            'image_classes': image_classes,
            'caption_classes': caption_classes,
        }

    @torch.no_grad()
    def retrieve(self, q_features, g_features,
                 q_ids, g_ids,
                 q_classes, g_classes,
                 topk=10, reduction='sum', batch_size=256):
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
        query_meta = {}

        for q_indices in self.pbar(batch(range(n_queries), batch_size=batch_size)):
            q_indices = np.array(q_indices)

            if self.eval_method != 'matching_prob':
                _q_feature = q_features[q_indices, :]
                _q_feature = _q_feature.view(len(q_indices) * self.n_embeddings, -1)
            else:
                _q_feature = q_features[q_indices, :, :]
            sims, pred_ranks = pmm(_q_feature, n_embeddings=self.n_embeddings, reduction=reduction)

            _, sorted_indices = pred_ranks.sort()
            for idx, sorted_db_index in enumerate(sorted_indices):
                _, _sorted_idx = sorted_db_index.sort()
                qidx = q_ids[q_indices[idx]]
                retrieved_items[qidx] = g_ids[to_numpy(_sorted_idx)[0][:topk]]
                retrieved_scores[qidx] = sims[idx][:topk].cpu().numpy()
                query_meta[qidx] = {'class': int(q_classes[q_indices[idx]].item()),
                                    'retrieved_classes': [int(item) for item in
                                                          g_classes[to_numpy(_sorted_idx)[0][:topk]].tolist()]}

        return retrieved_items, retrieved_scores, query_meta

    def get_scores(self, knn_labels, gt_labels, best_pred_ranks,
                   label_counts, embeddings_come_from_same_source,
                   sigmas=None, dismatch_probs=None):
        # https://github.com/KevinMusgrave/pytorch-metric-learning/blob/0b575b556fe339c2a62043d0ff0efe7fe85107bc/src/pytorch_metric_learning/utils/accuracy_calculator.py#L45
        relevance_mask = get_relevance_mask(
            knn_labels.shape, gt_labels, label_counts, embeddings_come_from_same_source
        )
        num_samples, num_k = knn_labels.shape
        equality = (knn_labels == gt_labels) * relevance_mask.astype(bool)
        cumulative_correct = np.cumsum(equality, axis=1)
        k_idx = np.tile(np.arange(1, num_k + 1), (num_samples, 1))
        precision_at_ks = (cumulative_correct * equality) / k_idx
        summed_precision_per_row = np.sum(precision_at_ks * relevance_mask, axis=1)
        max_possible_matches_per_row = np.sum(relevance_mask, axis=1)
        mean_average_precision_at_r = np.mean(summed_precision_per_row / max_possible_matches_per_row)

        pr_per_sample = r_precision(knn_labels, gt_labels, embeddings_come_from_same_source, label_counts)
        precision_r = np.mean(pr_per_sample)

        sigma_vs_map = None
        dismatchprob_vs_map = None

        recall_1 = recall_at_k(knn_labels, gt_labels, 1)
        recall_5 = recall_at_k(knn_labels, gt_labels, 5)
        recall_10 = recall_at_k(knn_labels, gt_labels, 10)

        precision_5 = precision_at_k(knn_labels, gt_labels, 5)
        precision_10 = precision_at_k(knn_labels, gt_labels, 10)
        precision_50 = precision_at_k(knn_labels, gt_labels, 50)
        precision_100 = precision_at_k(knn_labels, gt_labels, 100)

        scores = {
            'recall_1': recall_1,
            'recall_5': recall_5,
            'recall_10': recall_10,
            'rsum': recall_1 + recall_5 + recall_10,
            'precision_5': precision_5,
            'precision_10': precision_10,
            'precision_50': precision_50,
            'precision_100': precision_100,
            'r_precision': precision_r,
            'map_r': mean_average_precision_at_r,
        }

        if sigma_vs_map:
            scores['sigma_vs_map'] = sigma_vs_map
            scores['dismatchprob_vs_map'] = dismatchprob_vs_map

        if best_pred_ranks is not None:
            medr = np.floor(np.median(best_pred_ranks)) + 1
            meanr = np.mean(best_pred_ranks) + 1
            scores['medr'] = medr
            scores['meanr'] = meanr
        return scores

    @torch.no_grad()
    def evaluate_recall(self, q_features, g_features, q_labels, g_labels,
                        q_ids=None, g_ids=None,
                        q_classes=None, g_classes=None,
                        label_counts=None,
                        embeddings_come_from_same_source=False,
                        q_sigmas=None, g_sigmas=None,
                        reduction='sum', batch_size=256
                        ):
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
        else:
            pmm = HybridDistanceModule(self.eval_method)
        pmm.set_g_features(g_features, g_sigmas)

        q_features = q_features.to(self.eval_device)
        sigmas = None
        dismatch_probs = None
        if q_sigmas is not None and len(q_sigmas.shape) == 2:
            sigmas = []
            dismatch_probs = []

        knn_labels = []
        gt_labels = []
        for q_indices in self.pbar(batch(range(n_queries), batch_size=batch_size)):
            q_indices = np.array(q_indices)

            if self.eval_method != 'matching_prob':
                _q_feature = q_features[q_indices, :]
                _q_feature = _q_feature.view(len(q_indices) * self.n_embeddings, -1)
            else:
                _q_feature = q_features[q_indices, :, :]

            if q_sigmas is not None:
                _q_sigmas = q_sigmas[q_indices]
            else:
                _q_sigmas = None
            _, pred_ranks = pmm(_q_feature, n_embeddings=self.n_embeddings,
                                reduction=reduction, embeddings_come_from_same_source=embeddings_come_from_same_source,
                                q_indices=q_indices, q_sigmas=_q_sigmas)

            _, sorted_indices = pred_ranks.sort()
            for idx, sorted_db_index in enumerate(sorted_indices):
                _, _sorted_idx = sorted_db_index.sort()
                q_class = int(q_classes[q_indices[idx]].item())
                r_classes = [int(item) for item in
                             g_classes[to_numpy(_sorted_idx)[0]].tolist()]
                gt_labels.append(q_class)
                knn_labels.append(r_classes)

            for idx, q_idx in enumerate(q_indices):
                pos_indices = np.where(g_labels == q_labels[q_idx])[0]
                _pred_ranks = [torch.where(pred_ranks[idx] == pos_idx)[0][0].item() for pos_idx in pos_indices]
                best_pred_ranks[q_idx] = min(_pred_ranks)

        knn_labels = np.array(knn_labels)
        gt_labels = np.array(gt_labels)
        gt_labels = gt_labels[:, None]

        scores = self.get_scores(knn_labels, gt_labels, best_pred_ranks,
                                 label_counts, embeddings_come_from_same_source,
                                 sigmas, dismatch_probs)

        return scores

    @torch.no_grad()
    def evaluate(self, dataloader, n_crossfolds=None,
                 reduction='sum', eval_batch_size=256,
                 report_sigma=False,
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

        scores['image_sigma'] = np.mean(image_sigmas)
        scores['caption_sigma'] = np.mean(caption_sigmas)

        q_sigmas = None
        g_sigmas = None

        if report_sigma:
            q_sigmas = image_sigmas
            g_sigmas = caption_sigmas

        if self.logger:
            self.logger.log('evaluating i2t...')
        label_counts = {k: len(v) for k, v in dataloader.dataset.class_to_indices.items()}
        scores['i2t'] = self.evaluate_recall(image_features,
                                             caption_features,
                                             image_classes,
                                             caption_classes,
                                             image_classes,
                                             caption_classes,
                                             image_classes,
                                             caption_classes,
                                             label_counts,
                                             reduction=reduction,
                                             q_sigmas=q_sigmas,
                                             g_sigmas=g_sigmas,
                                             batch_size=eval_batch_size)

        if report_sigma:
            q_sigmas = caption_sigmas
            g_sigmas = image_sigmas

        if self.logger:
            self.logger.log('evaluating t2i...')
        label_counts = {k: len(v) for k, v in dataloader.dataset.class_to_img_indices.items()}
        scores['t2i'] = self.evaluate_recall(caption_features,
                                             image_features,
                                             caption_classes,
                                             image_classes,
                                             caption_classes,
                                             image_classes,
                                             caption_classes,
                                             image_classes,
                                             label_counts,
                                             reduction=reduction,
                                             q_sigmas=q_sigmas,
                                             g_sigmas=g_sigmas,
                                             batch_size=eval_batch_size)
        for key in ('rsum', 'medr', 'meanr'):
            scores[key] = scores['i2t'][key] + scores['t2i'][key]
        return scores
