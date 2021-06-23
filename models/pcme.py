""" PCME model base code

PCME
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import torch.nn as nn

from models.image_encoder import EncoderImage
from models.caption_encoder import EncoderText


class PCME(nn.Module):
    """Probabilistic CrossModal Embedding (PCME) module"""
    def __init__(self, word2idx, config):
        super(PCME, self).__init__()

        self.embed_dim = config.embed_dim
        if config.get('n_samples_inference', 0):
            self.n_embeddings = config.n_samples_inference
        else:
            self.n_embeddings = 1

        self.img_enc = EncoderImage(config)
        self.txt_enc = EncoderText(word2idx, config)

    def forward(self, images, sentences, lengths):
        image_output = self.img_enc(images)
        caption_output = self.txt_enc(sentences, lengths)

        return {
            'image_features': image_output['embedding'],
            'image_attentions': image_output.get('attention'),
            'image_residuals': image_output.get('residual'),
            'image_logsigma': image_output.get('logsigma'),
            'image_logsigma_att': image_output.get('uncertainty_attention'),
            'caption_features': caption_output['embedding'],
            'caption_attentions': caption_output.get('attention'),
            'caption_residuals': caption_output.get('residual'),
            'caption_logsigma': caption_output.get('logsigma'),
            'caption_logsigma_att': caption_output.get('uncertainty_attention'),
        }

    def image_forward(self, images):
        return self.img_enc(images)

    def text_forward(self, sentences, lengths):
        return self.txt_enc(sentences, lengths)
