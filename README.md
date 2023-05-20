# Probabilistic Cross-Modal Embedding (PCME) CVPR 2021

Official Pytorch implementation of PCME | [Paper](https://arxiv.org/abs/2101.05068)

[Sanghyuk Chun](https://sanghyukchun.github.io/home/)<sup>1</sup> [Seong Joon Oh](https://seongjoonoh.com/)<sup>1</sup> Rafael Sampaio de Rezende<sup>2</sup> [Yannis Kalantidis](https://www.skamalas.com/)<sup>2</sup> Diane Larlus<sup>2</sup>

<sup>1</sup><sub>[NAVER AI LAB](https://naver-career.gitbook.io/en/teams/clova-cic)</sub><br>
<sup>2</sup><sub>[NAVER LABS Europe](https://europe.naverlabs.com/)</sub>


<a href="https://www.youtube.com/watch?v=J_DaqSLEcVk"><img src="http://img.youtube.com/vi/J_DaqSLEcVk/0.jpg" 
alt="VIDEO" width="700" border="10" /></a>


## Updates

- 16 Jul, 2022: Add PCME CutMix-pretrained weight (used for [ECCV Caption](https://github.com/naver-ai/eccv-caption) paper)
- 23 Jun, 2021: Initial upload.

## Installation

Install dependencies using the following command.

```
pip install cython && pip install -r requirements.txt
python -c 'import nltk; nltk.download("punkt", download_dir="/opt/conda/nltk_data")'
git clone https://github.com/NVIDIA/apex && cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

### Dockerfile

You can use my docker image as well
```
docker pull sanghyukchun/pcme:torch1.2-apex-dali
```

Please Add `--model__cache_dir /vector_cache` when you run the code

## Configuration

All experiments are based on configuration files (see [config/coco](config/coco) and [config/cub](config/cub)).
If you want to change only a few options, instead of re-writing a new configuration file, you can override the configuration as the follows:

```
python <train | eval>.py --dataloader__batch_size 32 --dataloader__eval_batch_size 8 --model__eval_method matching_prob
```

See [config/parser.py](config/parser.py) for details

## Dataset preparation

### COCO Caption

We followed the same split provided by [VSE++](http://www.cs.toronto.edu/~faghri/vsepp/data.tar).
Dataset splits can be found in [datasets/annotations](datasets/annotations).

Note that we also need `instances_<train | val>2014.json` for computing PMRP score.

### CUB Caption

Download images (CUB-200-2011) from [this link](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), and download caption from [reedscot/cvpr2016](https://github.com/reedscot/cvpr2016).
You can use the image path and the caption path separately in the code.

## Evaluate pretrained models

NOTE: the current implementation of plausible match R-Precision (PMRP) is not efficient: <br>
It first dumps all ranked items for each item to a local file, and compute R-precision. <br>
We are planning to re-implement efficient PMRP as soon as possible.

### COCO Caption

```
# Compute recall metrics
python evaluate_recall_coco.py ./config/coco/pcme_coco.yaml \
    --dataset_root <your_dataset_path> \
    --model_path model_last.pth \
    # --model__cache_dir /vector_cache # if you use my docker image
```

```
# Compute plausible match R-Precision (PMRP) metric
python extract_rankings_coco.py ./config/coco/pcme_coco.yaml \
    --dataset_root <your_dataset_path> \
    --model_path model_last.pth \
    --dump_to <dumped_ranking_file> \
    # --model__cache_dir /vector_cache # if you use my docker image

python evaluate_pmrp_coco.py --ranking_file <dumped_ranking_file>
```

| Method   | I2T 1K PMRP | I2T 1K R@1 | I2T ECCV mAP@R | T2I 1K PMRP | T2I 1K R@1 | T2I ECCV mAP@R | Model file |
|----------|----------|---------|----------|----------|---------|----------|------------|
| PCME     | 45.0     | 68.8    |   26.2   | 46.0     | 54.6    |   48.0   | [link](https://github.com/naver-ai/pcme/releases/download/v1.0.0/pcme_coco.pth) |
| PCME (CutMix-pretrained) | 46.2 | 68.3 | 28.6 | 47.1 | 56.7 | 54.9 | [link](https://github.com/naver-ai/pcme/releases/download/v1.0.0/pcme_cutmix_coco.pth) |
| PVSE K=1 | 40.3     | 66.7    |   23.4   | 41.8     | 53.5    |   44.6   | -          |
| PVSE K=2 | 42.8     | 69.2    |   26.7   | 43.6     | 55.2    |   53.8   | -          |
| VSRN     | 41.2     | 76.2    |   30.8   | 42.4     | 62.8    |   53.8   | -          |
| VSRN + AOQ | 44.7   | 77.5    |   30.7   | 45.6     | 63.5    |   51.2   | -          |

Check [ECCV Caption dataset](https://github.com/naver-ai/eccv-caption) for more details of "ECCV mAP@R".
- Paper: [ECCV Caption: Correcting False Negatives by Collecting Machine-and-Human-verified Image-Caption Associations for MS-COCO](https://arxiv.org/abs/2204.03359)
- GitHub: [naver-ai/eccv-caption](https://github.com/naver-ai/eccv-caption)

### CUB Caption

```
python evaluate_cub.py ./config/cub/pcme_cub.yaml \
    --dataset_root <your_dataset_path> \
    --caption_root <your_caption_path> \
    --model_path model_last.pth \
    # --model__cache_dir /vector_cache # if you use my docker image
```

NOTE: If you just download file from [reedscot/cvpr2016](https://github.com/reedscot/cvpr2016), then `caption_root` will be `cvpr2016_cub/text_c10`

If you want to test other probabilistic distances, such as Wasserstein distance or KL-divergence, try the following command:

```
python evaluate_cub.py ./config/cub/pcme_cub.yaml \
    --dataset_root <your_dataset_path> \
    --caption_root <your_caption_path> \
    --model_path model_last.pth \
    --model__eval_method <distance_method> \
    # --model__cache_dir /vector_cache # if you use my docker image
```

You can choose `distance_method` in `['elk', 'l2', 'min', 'max', 'wasserstein', 'kl', 'reverse_kl', 'js', 'bhattacharyya', 'matmul', 'matching_prob']`


## How to train

NOTE: we train each model with mixed-precision training (O2) on a single V100.<br>
Since, the current code does not support multi-gpu training, if you use different hardware, the batchsize should be reduced.<br>
Please note that, hence, the results couldn't be reproduced if you use smaller hardware than V100.

### COCO Caption

```
python train_coco.py ./config/coco/pcme_coco.yaml --dataset_root <your_dataset_path> \
    # --model__cache_dir /vector_cache # if you use my docker image
```

It takes about 46 hours in a single V100 with mixed precision training.

### CUB Caption

We use CUB Caption dataset [(Reed, et al. 2016)](https://openaccess.thecvf.com/content_cvpr_2016/papers/Reed_Learning_Deep_Representations_CVPR_2016_paper.pdf) as a new cross-modal retrieval benchmark. Here, instead of matching the sparse paired image-caption pairs, we treat all image-caption pairs in the same class as **positive**. Since our split is based on the zero-shot learning benchmark [(Xian, et al. 2017)](https://openaccess.thecvf.com/content_cvpr_2017/papers/Xian_Zero-Shot_Learning_-_CVPR_2017_paper.pdf), we leave out 50 classes from 200 bird classes for the evaluation.

- Reed, Scott, et al. "Learning deep representations of fine-grained visual descriptions." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
- Xian, Yongqin, Bernt Schiele, and Zeynep Akata. "Zero-shot learning-the good, the bad and the ugly." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017.

#### hyperparameter search

We additionally use cross-validation splits by (Xian, et el. 2017), namely using 100 classes for training and 50 classes for validation. 

```
python train_cub.py ./config/cub/pcme_cub.yaml \
    --dataset_root <your_dataset_path> \
    --caption_root <your_caption_path> \
    --dataset_name cub_trainval1 \
    # --model__cache_dir /vector_cache # if you use my docker image
```

Similarly, you can use `cub_trainval2` and `cub_trainval3` as well.

#### training with full training classes

```
python train_cub.py ./config/cub/pcme_cub.yaml \
    --dataset_root <your_dataset_path> \
    --caption_root <your_caption_path> \
    # --model__cache_dir /vector_cache # if you use my docker image
```

It takes about 4 hours in a single V100 with mixed precision training.

## How to cite

```
@inproceedings{chun2021pcme,
    title={Probabilistic Embeddings for Cross-Modal Retrieval},
    author={Chun, Sanghyuk and Oh, Seong Joon and De Rezende, Rafael Sampaio and Kalantidis, Yannis and Larlus, Diane},
    year={2021},
    booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
}
```

## License

```
MIT License

Copyright (c) 2021-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
