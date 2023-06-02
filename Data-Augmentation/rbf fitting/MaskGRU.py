
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import argparse
import math
import utils

device = "cuda" if torch.cuda.is_available() else "cpu"
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

class mask_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, bias = True):
        super(mask_GRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.complement_layer = nn.Linear(hidden_size, input_size, bias = self.bias)
        self.reset_step = nn.Linear(input_size + hidden_size + input_size, hidden_size, bias = bias)
        # self.reset_step = nn.Linear(input_size, hidden_size, bias = bias)
        self.update_step = nn.Linear(input_size + hidden_size + input_size, hidden_size, bias = bias)
        self.hidden_step = nn.Linear(input_size + hidden_size + input_size, hidden_size, bias = bias)
        # self.sigmoid = F.sigmoid()
        # self.tanh = F.tanh()
        self.reset_parameters()

        self.temp_decay_h = TemporalDecay(input_size = self.input_size, output_size = self.hidden_size, diag = False)
        # self.temp_decay_x = TemporalDecay(input_size = self.input_size, output_size = self.hidden_size, diag = False)
    
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters(): 
            w.data.uniform_(-std, std)

    def init_hidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            hidden = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            return hidden
        else:
            hidden = Variable(torch.zeros(batch_size, self.hidden_size))
            return hidden

    def forward(self, x, mask, delta):

        hid = Variable(torch.zeros((x.size()[0], self.hidden_size))).to(device)
        
        x_loss = 0.0
        x_imputataions = []
        for t in range(x.size(1):
            
            value = x[:,t,:]
            m = mask[:,t,:]
            d = delta[:,t,:]

            gamma = self.temp_decay_h(d)
            hid = hid * gamma.detach()
            complement_x = self.complement_layer(hid)
            x_c = m * value  + (1- m) * complement_x        # no problem

            x_loss += torch.sum(torch.abs(value - complement_x) * m) / (torch.sum(m) + 1e-5)
            
            # GRU cell start
            combine = torch.cat([x_c, hid, m], dim = 1)
            reset = torch.sigmoid(self.reset_step(combine))
            update = torch.sigmoid(self.update_step(combine)) # sigmoid 때문은 아님
            
            h_c = torch.tanh(self.hidden_step(torch.cat([x_c, hid * reset, m], dim = 1)))

            hid = (1 - update.detach()) * hid + update.detach() * h_c.detach()
            x_imputataions.append(complement_x.unsqueeze(dim = 1))

        x_imputataions = torch.cat(x_imputataions, dim = 1)
            
            
        return complement_x, x_loss, x_imputataions

'''
input_size = 36
hidden_size = 108
seq_len = len(df)
data_iter = get_loader(df, df_ground)
model2 = mask_GRU(input_size, hidden_size, seq_len).to(device)
hid = None
epochs = 3000
torch.autograd.set_detect_anomaly(False)
optimizer = optim.Adam(model2.parameters(), lr=0.001)
model2.train()
progress = tqdm(range(epochs))

imputation_list = []

for epoch in progress:
    batch_loss = 0.0
    imputations = []
    for values, masks, evals, eval_masks, deltas in data_iter:
        values = values.to(device)
        masks = masks.to(device)
        deltas = deltas.to(device)
        hid, complement_x, x_loss, x_imputation = model2(values, masks, deltas)
        batch_loss += x_loss
        
    optimizer.zero_grad()
    x_loss.backward()
    optimizer.step()
    imputation_list.append(x_imputation)
    progress.set_description("loss: {:0.3f}".format(x_loss/len(data_iter)))

'''