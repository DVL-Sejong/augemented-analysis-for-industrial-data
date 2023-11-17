import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from tqdm import tqdm
import math
import random
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.autograd import profiler
from torch.nn.parallel import DataParallel

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device = "cuda" if torch.cuda.is_available() else "cpu"

class FeatureRegression(nn.Module):
    def __init__(self, input_size):
        super(FeatureRegression, self).__init__()
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * Variable(self.m), self.b)
        return z_h

class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag = False):
        super(TemporalDecay, self).__init__()
        self.diag = diag

        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))
        self.relu = nn.ReLU(inplace=False)
        if self.diag == True:
            assert(input_size == output_size)
            m = torch.eye(input_size, input_size)
            self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        gamma = self.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma

class MGRU_ori(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MGRU_ori, self).__init__()

        self.temp_decay_h = TemporalDecay(input_size, output_size = hidden_size, diag = False)
        self.temp_decay_x = TemporalDecay(input_size, input_size, diag = True)
        self.temp_decay_r = TemporalDecay(input_size, input_size, diag = True)
        
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.build()

    def build(self):
        self.output_layer = nn.Linear(self.hidden_size, self.input_size, bias=True)
        
        self.z_layer = FeatureRegression(self.input_size)
        self.beta_layer = nn.Linear(self.input_size * 2, self.input_size)
        self.grucell = nn.GRUCell(self.input_size * 2, self.hidden_size)
        self.concat_lyaer = nn.Linear(self.input_size * 2, self.input_size)
        

    def loss(self, hat, y, m):
        return torch.sum(torch.abs((y - hat)) * m) / (torch.sum(m) + 1e-5)

    
    def forward(self, input):
        values = input[:,0,::]
        delta = input[:,1,::]
        masks = input[:,2,::]
        #rbfs = input[:,3,::]

        hid = torch.zeros((values.size(0), self.hidden_size)).to(input.device)

        x_loss = 0.0
        imputations = []
        c_hat_list = []
        for i in range(values.size(1)):

            v = values[:,i,:]
            d = delta[:,i,:]
            m = masks[:,i,:]
            # r = rbfs[:,i,:]

            gamma_x = self.temp_decay_x(d)
            gamma_h = self.temp_decay_h(d)
            
            hid = hid * gamma_h

            
            x_hat = self.output_layer(hid)
            x_loss += torch.sum(torch.abs(v - x_hat) * m) / (torch.sum(m) + 1e-5)

            x_c = m * v + (1 - m) * x_hat

            z_hat = self.z_layer(x_c)
            x_loss += torch.sum(torch.abs(v - z_hat) * m) / (torch.sum(m) + 1e-5)

            beta_weight = torch.cat([gamma_x, m], dim = 1)
            beta = self.beta_layer(beta_weight)

            c_hat = beta * z_hat + (1 - beta) * x_hat
            x_loss += torch.sum(torch.abs(v - c_hat) * m) / (torch.sum(m) + 1e-5)

            c_c = m * v + (1 - m) * c_hat

            gru_input = torch.cat([c_c, m], dim = 1)
            imputations.append(c_c.unsqueeze(dim = 1))
            c_hat_list.append(c_hat.unsqueeze(1))
            
            # GRU cell
            hid = self.grucell(gru_input, hid)

        c_hat_list = torch.cat(c_hat_list, dim = 1)
        imputations = torch.cat(imputations, dim = 1)
        return c_hat_list, imputations, x_loss

class BiMGRU_ori(nn.Module):
    def __init__(self, input_size, hidden_size, bias = True):
        super(BiMGRU_ori, self).__init__()
        
        self.fmGRU = MGRU_ori(input_size, hidden_size)
        self.bmGRU = MGRU_ori(input_size, hidden_size)
    
    def get_consistency_loss(self, pred_f, pred_b):
        loss = torch.pow(pred_f - pred_b, 2.0).mean()
        return loss
    
    def backdirect_data(self, tensor_):
        if tensor_.dim() <= 1:
            return tensor_
        # print(tensor_.device)
        indices = range(tensor_.size()[2])[::-1]
        indices = Variable(torch.LongTensor(indices), requires_grad = False).to(tensor_.device)


        return tensor_.index_select(2, indices)
    
    def backdirect_imputation(self, tensor_):
        if tensor_.dim() <= 1:
            return tensor_
        # print(self.device)
        indices = range(tensor_.size()[1])[::-1]
        indices = Variable(torch.LongTensor(indices), requires_grad = False).to(tensor_.device)

        return tensor_.index_select(1, indices)

    def forward(self, dataset):


        c_hat_list, imputations, x_loss = self.fmGRU(dataset)
        back_dataset = self.backdirect_data(dataset)
        back_c_hat_list, back_x_imputataions, back_x_loss = self.bmGRU(back_dataset)

        loss_c = self.get_consistency_loss(c_hat_list, self.backdirect_imputation(back_c_hat_list))
        loss = x_loss + back_x_loss + loss_c

        bi_c_hat = (c_hat_list +  self.backdirect_imputation(back_c_hat_list)) / 2
        bi_imputation = (imputations +  self.backdirect_imputation(back_x_imputataions)) / 2
        # print(loss)
        return loss, x_loss, back_x_loss, loss_c, bi_c_hat, bi_imputation

def train_BiMGRU(model, lr, epochs, dataset, device):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    progress = tqdm(range(epochs))
    
    # imputation_list = []
    # loss_list = []
    model = DataParallel(model, output_device=0)
    model.to('cuda')


    model.train()
    
    for epoch in progress:
        batch_loss = 0.0
        batch_f_loss = 0.0
        batch_b_loss = 0.0
        batch_c_loss = 0.0

        for data in dataset:
            data = data.to('cuda')

            loss, x_loss, back_x_loss, loss_c, bi_chat, biimputataion = model(data)

            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
            # imputation_list.append(bi_chat)

            batch_loss += loss.sum()
            batch_f_loss += x_loss.sum()
            batch_c_loss += loss_c.sum()
            batch_b_loss += back_x_loss.sum()
            
        progress.set_description("loss: {}, f_MGRU loss : {}, b_MGRU loss : {}, consistency Loss : {}".format(batch_loss, batch_f_loss, batch_b_loss, batch_c_loss))
            
    
'''
class FCN_Regression(nn.Module):
    def __init__(self, feature_num, rnn_hid_size):
        super(FCN_Regression, self).__init__()
        self.feat_reg = FeatureRegression(rnn_hid_size * 2)
        self.U = Parameter(torch.Tensor(feature_num, feature_num))
        self.V1 = Parameter(torch.Tensor(feature_num, feature_num))
        self.V2 = Parameter(torch.Tensor(feature_num, feature_num))
        self.beta = Parameter(torch.Tensor(feature_num))  # bias beta
        self.final_linear = nn.Linear(feature_num, feature_num)

        m = torch.ones(feature_num, feature_num) - torch.eye(feature_num, feature_num)
        self.register_buffer("m", m)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.U.size(0))
        self.U.data.uniform_(-stdv, stdv)
        self.V1.data.uniform_(-stdv, stdv)
        self.V2.data.uniform_(-stdv, stdv)
        self.beta.data.uniform_(-stdv, stdv)

    def forward(self, x_t, m_t, target):
        h_t = torch.tanh(
            F.linear(x_t, self.U * self.m)
            + F.linear(target, self.V1 * self.m)
            + F.linear(m_t, self.V2)
            + self.beta
        )
        x_hat_t = self.final_linear(h_t)
        return x_hat_t

class MRNN(nn.Module):
    def __init__(self, input_size, rnn_hid_size):
        super(MRNN, self).__init__()

        self.input_size = input_size
        self.rnn_hid_size = rnn_hid_size

        self.build()

    def build(self):
        self.rnn_cell_f = nn.GRUCell(self.input_size * 3, self.rnn_hid_size)
        self.rnn_cell_b = nn.GRUCell(self.input_size * 3, self.rnn_hid_size)
        self.pred_rnn = nn.LSTM(self.input_size, self.rnn_hid_size, batch_first = True)

        self.temp_decay_h = TemporalDecay(input_size = self.input_size, output_size = self.rnn_hid_size, diag = False)
        self.temp_decay_x = TemporalDecay(input_size = self.input_size, output_size = self.input_size, diag = True)

        self.concated_hidden_project = nn.Linear(self.rnn_hid_size * 2, self.input_size)
        self.fcn_regression  = FCN_Regression(self.input_size, self.rnn_hid_size)

        self.weight_combine = nn.Linear(self.input_size * 2, self.input_size)

        self.dropout = nn.Dropout(p = 0.25)
        self.out = nn.Linear(self.rnn_hid_size, 1)
    

    def get_hidden_f(self, input):
        values = input[:,0,::]
        deltas = input[:,1,::]
        masks = input[:,2,::]

        hiddens = []

        h = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))

        if torch.cuda.is_available():
            h= h.cuda()

        for t in range(values.size(1)):
        

            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]

            inputs = torch.cat([x, m, d], dim = 1)

            h = self.rnn_cell_f(inputs, h)

            hiddens.append(h)

        return hiddens
    
    def get_hidden_b(self, input):
        values = input[:,0,::]
        deltas = input[:,1,::]
        masks = input[:,2,::]

        hiddens = []

        h = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))

        if torch.cuda.is_available():
            h = h.cuda()

        for t in range(values.size(1)):
            

            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]

            inputs = torch.cat([x, m, d], dim = 1)

            h = self.rnn_cell_b(inputs, h)

            hiddens.append(h)

        return hiddens


    def forward(self, input):
        # Original sequence with 24 time steps
        hidden_forward = self.get_hidden_f(input)
        hidden_backward = self.get_hidden_b(input)[::-1]

        values = input[:,0,::]
        deltas = input[:,1,::]
        masks = input[:,2,::]

        x_loss = 0.0

        imputations = []

        for t in range(values.size(1)):
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]

            hf = hidden_forward[t]
            hb = hidden_backward[t]
            h = torch.cat([hf, hb], dim = 1)

            RNN_estimation  = self.concated_hidden_project(h)
            RNN_imputed_data = m * x + (1 - m) * RNN_estimation

            x_loss += torch.sum(torch.abs(x - RNN_estimation) * m) / (torch.sum(m) + 1e-5)

            FCN_estimation  = self.fcn_regression(x, m, RNN_imputed_data)

            x_loss += torch.sum(torch.abs(x - FCN_estimation) * m) / (torch.sum(m) + 1e-5)

            imputations.append(FCN_estimation.unsqueeze(dim = 1))

        imputations = torch.cat(imputations, dim = 1)

        return imputations, x_loss * 5
''' 

'''
def train_MRNN(model, lr, epochs, dataset, device):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    progress = tqdm(range(epochs))
    
    model.to(device)
    dataset.to(device)
    for epoch in progress:
        batch_loss = 0.0

        for data in dataset:
            # data = data.to(device)
            imputations, x_loss  = model(data)
        
            optimizer.zero_grad()
            x_loss.backward()
            optimizer.step()
            
            batch_loss += x_loss


        progress.set_description("loss: {}".format(batch_loss))

    return x_loss
    '''