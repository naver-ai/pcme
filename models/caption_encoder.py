""" Caption encoder based on PVSE implementation.
Reference code:
    https://github.com/yalesong/pvse/blob/master/model.py
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torchtext

from models.pie_model import PIENet
from models.uncertainty_module import UncertaintyModuleText
from utils.tensor_utils import l2_normalize, sample_gaussian_tensors


def get_pad_mask(max_length, lengths, set_pad_to_one=True):
    ind = torch.arange(0, max_length).unsqueeze(0).to(lengths.device)
    mask = (ind >= lengths.unsqueeze(1)) if set_pad_to_one \
        else (ind < lengths.unsqueeze(1))
    mask = mask.to(lengths.device)
    return mask


class EncoderText(nn.Module):
    def __init__(self, word2idx, opt):
        super(EncoderText, self).__init__()

        wemb_type, word_dim, embed_dim = \
            opt.wemb_type, opt.word_dim, opt.embed_dim

        self.embed_dim = embed_dim
        self.use_attention = opt.txt_attention
        self.use_probemb = opt.get('txt_probemb')

        # Word embedding
        self.embed = nn.Embedding(len(word2idx), word_dim)
        self.embed.weight.requires_grad = opt.txt_finetune

        # Sentence embedding
        self.rnn = nn.GRU(word_dim, embed_dim // 2, bidirectional=True, batch_first=True)

        if self.use_attention:
            self.pie_net = PIENet(1, word_dim, embed_dim, word_dim // 2)

        self.uncertain_net = UncertaintyModuleText(word_dim, embed_dim, word_dim // 2)
        self.init_weights(wemb_type, word2idx, word_dim, opt.cache_dir)

        self.n_samples_inference = opt.get('n_samples_inference', 0)

    def init_weights(self, wemb_type, word2idx, word_dim, cache_dir):
        if wemb_type is None:
            nn.init.xavier_uniform_(self.embed.weight)
        else:
            # Load pretrained word embedding
            if 'fasttext' == wemb_type.lower():
                wemb = torchtext.vocab.FastText(cache=cache_dir)
            elif 'glove' == wemb_type.lower():
                wemb = torchtext.vocab.GloVe(cache=cache_dir)
            else:
                raise Exception('Unknown word embedding type: {}'.format(wemb_type))
            assert wemb.vectors.shape[1] == word_dim

            # quick-and-dirty trick to improve word-hit rate
            missing_words = []
            for word, idx in word2idx.items():
                if word not in wemb.stoi:
                    word = word.replace('-', '').replace('.', '').replace("'", '')
                    if '/' in word:
                        word = word.split('/')[0]
                if word in wemb.stoi:
                    self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
                else:
                    missing_words.append(word)
            print('Words: {}/{} found in vocabulary; {} words missing'.format(
                len(word2idx) - len(missing_words), len(word2idx), len(missing_words)))

    def forward(self, x, lengths):
        # Embed word ids to vectors
        wemb_out = self.embed(x)

        # Forward propagate RNNs
        packed = pack_padded_sequence(wemb_out, lengths, batch_first=True)
        if torch.cuda.device_count() > 1:
            self.rnn.flatten_parameters()
        rnn_out, _ = self.rnn(packed)
        padded = pad_packed_sequence(rnn_out, batch_first=True)

        # Reshape *final* output to (batch_size, hidden_size)
        I = lengths.expand(self.embed_dim, 1, -1).permute(2, 1, 0) - 1
        out = torch.gather(padded[0], 1, I).squeeze(1)

        output = {}

        if self.use_attention:
            pad_mask = get_pad_mask(wemb_out.shape[1], lengths, True)
            out, attn, residual = self.pie_net(out, wemb_out, pad_mask)
            output['attention'] = attn
            output['residual'] = residual

        if self.use_probemb:
            if not self.use_attention:
                pad_mask = get_pad_mask(wemb_out.shape[1], lengths, True)
            uncertain_out = self.uncertain_net(wemb_out, pad_mask, lengths)
            logsigma = uncertain_out['logsigma']
            output['logsigma'] = logsigma
            output['uncertainty_attention'] = uncertain_out['attention']

        out = l2_normalize(out)

        if self.use_probemb and self.n_samples_inference:
            output['embedding'] = sample_gaussian_tensors(out, logsigma, self.n_samples_inference)
        else:
            output['embedding'] = out

        return output
