from __future__ import absolute_import

from PIL import Image
import numpy as np
import sys
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from encoder import ResNet_ASTER
from decoder import AttentionRecognitionHead 


class ModelBuilder(nn.Module):
    def __init__(self, rec_num_classes, sDim, attDim, max_len_labels):  
        super(ModelBuilder, self).__init__()
        self.rec_num_classes = rec_num_classes
        self.sDim = sDim
        self.attDim = attDim
        self.max_len_labels = max_len_labels

        self.encoder = res_model     
        encoder_out_planes = self.encoder.out_planes
        self.decoder = AttentionRecognitionHead(num_classes=rec_num_classes,
                                                in_planes=512,
                                                sDim=sDim,
                                                attDim=attDim,
                                                max_len_labels=max_len_labels)
        self.criterion = nn.NLLLoss(ignore_index = 1832, reduction='mean')
        # 1832 is the index with padding in voc

    def forward(self, x, targets, lengths):

        batch_size = x.size(0)
        
        encoder_features, rnn_hidden = self.encoder(x)
        encoder_features = encoder_features.contiguous()
        rec_pred = self.decoder(encoder_features, rnn_hidden)
      
        rec_pred = rec_pred.transpose(0,1)
        rec_pred = rec_pred.contiguous()
        loss_rec = self.criterion(rec_pred.view(batch_size*17,-1), targets.view(-1))
            
        return loss_rec, rec_pred
       
