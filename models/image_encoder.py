""" Image encoder based on PVSE implementation.
Reference code:
    https://github.com/yalesong/pvse/blob/master/model.py
"""
import torch.nn as nn
from torchvision import models

from models.pie_model import PIENet
from models.uncertainty_module import UncertaintyModuleImage
from utils.tensor_utils import l2_normalize, sample_gaussian_tensors


class EncoderImage(nn.Module):
    def __init__(self, config):
        super(EncoderImage, self).__init__()

        embed_dim = config.embed_dim
        self.use_attention = config.img_attention
        self.use_probemb = config.get('img_probemb')

        # Backbone CNN
        self.cnn = getattr(models, config.cnn_type)(pretrained=True)
        cnn_dim = self.cnn_dim = self.cnn.fc.in_features

        self.avgpool = self.cnn.avgpool
        self.cnn.avgpool = nn.Sequential()

        self.fc = nn.Linear(cnn_dim, embed_dim)

        self.cnn.fc = nn.Sequential()

        if self.use_attention:
            self.pie_net = PIENet(1, cnn_dim, embed_dim, cnn_dim // 2)

        if self.use_probemb:
            self.uncertain_net = UncertaintyModuleImage(cnn_dim, embed_dim, cnn_dim // 2)

        for idx, param in enumerate(self.cnn.parameters()):
            param.requires_grad = config.img_finetune

        self.n_samples_inference = config.get('n_samples_inference', 0)

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, images):
        out_7x7 = self.cnn(images).view(-1, self.cnn_dim, 7, 7)
        pooled = self.avgpool(out_7x7).view(-1, self.cnn_dim)
        out = self.fc(pooled)

        output = {}
        out_7x7 = out_7x7.view(-1, self.cnn_dim, 7 * 7)

        if self.use_attention:
            out, attn, residual = self.pie_net(out, out_7x7.transpose(1, 2))
            output['attention'] = attn
            output['residual'] = residual

        if self.use_probemb:
            uncertain_out = self.uncertain_net(pooled, out_7x7.transpose(1, 2))
            logsigma = uncertain_out['logsigma']
            output['logsigma'] = logsigma
            output['uncertainty_attention'] = uncertain_out['attention']

        out = l2_normalize(out)

        if self.use_probemb and self.n_samples_inference:
            output['embedding'] = sample_gaussian_tensors(out, logsigma, self.n_samples_inference)
        else:
            output['embedding'] = out

        return output
