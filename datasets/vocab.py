""" Create a vocabulary wrapper.

Original code:
https://github.com/yalesong/pvse/blob/master/vocab.py
"""

from collections import Counter
import json
import os
import pickle

import fire
from nltk.tokenize import word_tokenize
from pycocotools.coco import COCO

ANNOTATIONS = {
    'mrw': ['mrw-v1.0.json'],
    'tgif': ['tgif-v1.0.tsv'],
    'coco': ['annotations/captions_train2014.json',
             'annotations/captions_val2014.json'],
}


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.idx = 0
        self.word2idx = {}
        self.idx2word = {}

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def load_from_pickle(self, data_path):
        with open(data_path, 'rb') as fin:
            data = pickle.load(fin)
        self.idx = data['idx']
        self.word2idx = data['word2idx']
        self.idx2word = data['idx2word']

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def from_tgif_tsv(path):
    captions = [line.strip().split('\t')[1]
                for line in open(path, 'r').readlines()]
    return captions


def from_mrw_json(path):
    dataset = json.load(open(path, 'r'))
    captions = []
    for datum in dataset:
        cap = datum['sentence']
        cap = cap.replace('/r/', '')
        cap = cap.replace('r/', '')
        cap = cap.replace('/u/', '')
        cap = cap.replace('u/', '')
        cap = cap.replace('..', '')
        cap = cap.replace('/', ' ')
        cap = cap.replace('-', ' ')
        captions += [cap]
    return captions


def from_coco_json(path):
    coco = COCO(path)
    ids = coco.anns.keys()
    captions = []
    for idx in ids:
        captions.append(str(coco.anns[idx]['caption']))

    return captions


def from_txt(txt):
    captions = []
    with open(txt, 'rb') as f:
        for line in f:
            captions.append(line.strip())
    return captions


def build_vocab(data_path, data_name, jsons, threshold):
    """Build a simple vocabulary wrapper."""
    counter = Counter()
    for path in jsons[data_name]:
        full_path = os.path.join(os.path.join(data_path, data_name), path)
        if data_name == 'tgif':
            captions = from_tgif_tsv(full_path)
        elif data_name == 'mrw':
            captions = from_mrw_json(full_path)
        elif data_name == 'coco':
            captions = from_coco_json(full_path)
        else:
            captions = from_txt(full_path)

        for caption in captions:
            tokens = word_tokenize(caption.lower())
            counter.update(tokens)

    # Discard if the occurrence of the word is less than min_word_cnt.
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    print('Vocabulary size: {}'.format(len(words)))

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add words to the vocabulary.
    for word in words:
        vocab.add_word(word)
    return vocab


def main(data_path, data_name, threshold=0):
    vocab = build_vocab(data_path, data_name, jsons=ANNOTATIONS, threshold=threshold)
    if not os.path.isdir('./vocab'):
        os.makedirs('./vocab')
    with open('./vocab/%s_vocab.pkl' % data_name, 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
    print("Saved vocabulary file to ", './vocab/%s_vocab.pkl' % data_name)


if __name__ == '__main__':
    fire.Fire(main)
