import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import numpy as np
from copy import deepcopy

device = "cuda" if torch.cuda.is_available() else "cpu"

# CNN 경우 window 크기가 변경될시 안에 들어가는 strid padding kernel 등 모두 바꿔줘야됨.
# 지금은 내가 임의로 맞춰놓은것.

class CNNEEG(nn.Module):
    def __init__(self, input_channel, keep_batch_dim=True):
        super(CNNEEG, self).__init__()

        self.input_channel = input_channel  # input_feature_num
        self.keep_batch_dim = keep_batch_dim
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.input_channel, self.input_channel, 8, stride=2, padding=3, groups=self.input_channel),
            nn.Conv1d(self.input_channel, 128, kernel_size=1))
        self.conv2 = nn.Sequential(nn.Conv1d(128, 128, 8, stride=4, padding=4, groups=128),
                                   nn.Conv1d(128, 64, kernel_size=1))
        self.conv3 = nn.Sequential(nn.Conv1d(64, 64, 8, stride=10, padding=0, groups=64),
                                   nn.Conv1d(64, self.input_channel, kernel_size=1))

        self.fc = nn.Linear(self.input_channel, 1)
        self.network = nn.Sequential(self.conv1,
                                     self.conv2,
                                     self.conv3,
                                     )

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def Flatten(self, data):
        if self.keep_batch_dim:
            return data.view(data.size(0), -1)
        else:
            return data.view(-1)

    def forward(self, X):

        pred = self.network(X)
        pred = self.fc(self.Flatten(pred))
        pred = self.sigmoid(pred)

        return pred


def restore_parameters(model, best_model):
    '''Move parameter values from best_model to model.'''
    for params, best_params in zip(model.parameters(), best_model.parameters()):
        params.data = best_params


def train_CNNEEG(model, trainloader, epochs, lr, device):
    model.to(device)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {'loss': []}

    best_it = None
    best_loss = np.inf
    best_model = None

    for epoch in range(epochs):
        losses = []
        for X, Y in trainloader:
            model.zero_grad()
            pred = model(X).squeeze(1)
            loss = loss_fn(pred, Y.float())

            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        avg_loss = np.mean(losses)
        history['loss'].append(avg_loss)
        print("Epoch {} / {}: Loss = {:.3f}".format(epoch + 1, epochs, avg_loss))

        if best_loss > avg_loss:
            best_it = epoch
            best_loss = avg_loss
            best_model = deepcopy(model)

    print('best epoch :{} epoch'.format(best_it))

    restore_parameters(model, best_model)

    return history

'''
data = torch.randn((100, 170,128), requires_grad = True, device = device) #(data 총 개수, feature 개수, window 크기)
target_ = torch.randint(0, 2, (100,), device = device)

trainloader = DataLoader(TensorDataset(data, target_), 
                         batch_size=50, 
                         shuffle=True)
                         
model = CNNEEG(170)
his = train_CNNEEG(model, trainloader, 10, 0.01, device = device)
'''