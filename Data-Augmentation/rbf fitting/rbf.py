import torch
import torch.nn as nn

def rbf_gaussian(self, input_data):
        out = torch.exp(-1 *(torch.pow((input_data - self.centers), 2))) / (torch.pow(self.sigma, 2))

        return out

def forward(self, input_data):
        R = self.rbf_gaussian(input_data)
        pred = torch.mm(self.weights, R).reshape(self.in_feature, 1, input_data.size(-1))

        return R, pred