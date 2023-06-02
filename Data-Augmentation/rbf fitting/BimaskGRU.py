import pandas as pd
import numpy as np
import torch
import torch.nn as 
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from MaskGRU import mask_GRU
import argparse
import math
import utils

device = "cuda" if torch.cuda.is_available() else "cpu"

class bimask_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, bias = True):
        super(bimask_GRU, self).__init__()
        
        self.fmGRU = mask_GRU(input_size, hidden_size, bias = bias)
        self.bmGRU = mask_GRU(input_size, hidden_size, bias = bias)
    
    def get_consistency_loss(self, pred_f, pred_b):
        loss = torch.pow(pred_f - pred_b, 2.0).mean()
        return loss
    
    def backdirect_data(self, tensor_):
        if tensor_.dim() <= 1:
            return tensor_
        indices = range(tensor_.size()[1])[::-1]
        indices = Variable(torch.LongTensor(indices), requires_grad = False)

        if torch.cuda.is_available():
            indices = indices.cuda()

        return tensor_.index_select(1, indices)

    def forward(self, x, mask, delta):
        back_x = self.backdirect_data(x)
        back_mask = self.backdirect_data(mask)
        back_delta = self.backdirect_data(delta)

        complement_x, x_loss, x_imputataions = self.fmGRU(x, mask, delta)
        back_complement_x, back_x_loss, back_x_imputataions = self.bmGRU(back_x, back_mask, back_delta)

        loss_c = self.get_consistency_loss(complement_x, back_complement_x)
        loss = x_loss + back_x_loss + loss_c

        biimputataion = (x_imputataions + back_x_imputataions) / 2

        return  loss, biimputataion
