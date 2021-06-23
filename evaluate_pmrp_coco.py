"""
PCME
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import fire
import os

import torch
import numpy as np
import ujson as json
from tqdm import tqdm
from datasets.coco import CocoCaptionsCap


def mean_score(score_dict):
    return 100 * np.mean([list(l) for l in score_dict.values()])


def evaluate_plausible_metrics(data, modal, pdists, iid_to_idx, val_image_ids, max_thres=3):
    rp_per_query_per_amb = {}
    r1_per_query_per_amb = {}

    for thres in tqdm(range(max_thres)):
        rp_per_query = []
        r1_per_query = []
        for key, _data in tqdm(data.items()):
            if modal == 'image':
                iid = key
            elif modal == 'text':
                iid = _data['query']['image_id']
            if iid not in val_image_ids:
                continue
            image_id_key = 'image_id' if modal == 'image' else 'id'
            distance = pdists[iid_to_idx[iid]]
            retrieved_sim = np.array([distance[iid_to_idx[_d[image_id_key]]]
                                      for _d in _data['retrieved']
                                      if _d[image_id_key] in val_image_ids])

            matched = retrieved_sim <= thres
            R = np.sum(matched)
            rp = np.sum(matched[:R]) / R
            rp_per_query.append(rp)
            r1_per_query.append(int(matched[0]))
        rp_per_query_per_amb[thres] = rp_per_query
        r1_per_query_per_amb[thres] = r1_per_query

    return rp_per_query_per_amb, r1_per_query_per_amb


def prepare_data(dataset_root, ids, n_fold_idx, ds=None):
    root = os.path.join(dataset_root, 'images/trainval35k')
    annotation_path = os.path.join(dataset_root, 'annotations/annotations/captions_val2014.json')
    instance_annotation_path = os.path.join(dataset_root, 'annotations/annotations/instances_val2014.json')

    if ds is None:
        ds = CocoCaptionsCap(root, annotation_path, ids=ids, instance_annFile=instance_annotation_path)
    if n_fold_idx > -1:
        image_ids = []
        seen_iid = set()
        for idx in tqdm(range(len(ds)), total=len(ds)):
            _, _, _, image_id = ds[idx]
            if image_id in seen_iid:
                continue
            seen_iid.add(image_id)
            image_ids.append(image_id)
        N = len(image_ids) // 5
        all_image_ids = np.array(image_ids)[n_fold_idx * N: (n_fold_idx + 1) * N]
        all_image_ids = set(all_image_ids)
    else:
        all_image_ids = ds.all_image_ids

    with open(instance_annotation_path) as fin:
        instance_ann = json.load(fin)

    iid_to_cls = {}
    for ann in tqdm(instance_ann['annotations']):
        image_id = int(ann['image_id'])
        if image_id not in all_image_ids:
            continue
        code = iid_to_cls.get(image_id, [0] * 90)
        code[int(ann['category_id']) - 1] = 1
        iid_to_cls[image_id] = code

    iid_to_codes = np.zeros((len(iid_to_cls), 90))
    val_image_ids = list(iid_to_cls.keys())
    for idx, _id in enumerate(val_image_ids):
        iid_to_codes[idx] = iid_to_cls[_id]

    N = iid_to_codes.shape[0]
    pdists = np.zeros((N, N))
    for idx, code in tqdm(enumerate(iid_to_codes)):
        pdists[idx] = np.sum(np.abs(iid_to_codes - code), axis=1)

    iid_to_idx = {iid: idx for idx, iid in enumerate(val_image_ids)}
    return ds, iid_to_codes, val_image_ids, pdists, iid_to_idx


def main(ranking_fname, dataset_root, n_fold=None, max_thres=3):
    data = torch.load(ranking_fname)
    te_ids = np.load('./datasets/annotations/coco_test_ids.npy')
    if n_fold:
        n_folds = [0, 1, 2, 3, 4]
    else:
        n_folds = [-1]

    i2t_pmrps, i2t_pmr1s = [], []
    t2i_pmrps, t2i_pmr1s = [], []
    ds = None
    for n_fold_idx in n_folds:
        ds, iid_to_codes, val_image_ids, pdists, iid_to_idx = prepare_data(dataset_root, te_ids, n_fold_idx, ds=ds)

        _i2t_pmrp, _i2t_pmr1 = \
            evaluate_plausible_metrics(data['i2t'], 'image', pdists, iid_to_idx, set(val_image_ids), max_thres=max_thres)
        _t2i_pmrp, _t2i_pmr1 = \
            evaluate_plausible_metrics(data['t2i'], 'text', pdists, iid_to_idx, set(val_image_ids), max_thres=max_thres)

        i2t_pmrps.append(mean_score(_i2t_pmrp))
        i2t_pmr1s.append(mean_score(_i2t_pmr1))
        t2i_pmrps.append(mean_score(_t2i_pmrp))
        t2i_pmr1s.append(mean_score(_t2i_pmr1))
    print('------------------------------')
    print(f'image-to-text PMRP: {np.mean(i2t_pmrps)}')
    print(f'text-to-image PMRP: {np.mean(t2i_pmrps)}')


if __name__ == '__main__':
    fire.Fire(main)
