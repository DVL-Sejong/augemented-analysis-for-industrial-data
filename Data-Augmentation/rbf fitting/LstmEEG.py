import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim  as optim
import numpy as np
from copy import deepcopy

device = "cuda" if torch.cuda.is_available() else "cpu"



class LSTMEEG(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMEEG, self).__init__()

        self.input_dim = input_dim     # feature 개수
        self.hidden_dim = hidden_dim   # 아무렇게나

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def init_hidden(self, batch):
        '''Initialize hidden states for LSTM cell.'''
        device = self.lstm.weight_ih_l0.device
        return (torch.zeros(1, batch, self.hidden_dim, device=device),
                torch.zeros(1, batch, self.hidden_dim, device=device))

    def forward(self, X, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(X.shape[0])

        lstm_out, (h, c) = self.lstm(X, hidden)
        y = self.fc(lstm_out[:, -1])
        y = self.sigmoid(y)  # binary classification
        # y = self.softmax(y) # multi-class #만약 label 이 3개 이상이면 위에거 주석처리하고 이거 주석풀어

        return y

def restore_parameters(model, best_model): # best model 저장하는 코드
    '''Move parameter values from best_model to model.'''
    for params, best_params in zip(model.parameters(), best_model.parameters()):
        params.data = best_params

def train_LSTMEEG(model, trainloader, epochs, lr, device):
    model.to(device)
    loss_fn = nn.BCELoss()   # loss function 바꿔도 됨.
    optimizer = optim.Adam(model.parameters(), lr=lr) # 바꿔도됨
    history = {'loss': []}

    best_it = None
    best_loss = np.inf
    best_model = None

    for epoch in range(epochs):
        losses = []
        for X, Y in trainloader:  # batch
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
# Example 
data = torch.randn((100, 30, 170), requires_grad = True, device = device) # (data 총 개수, window 크기, feature 개수)
target_ = torch.randint(0, 2, (100,), device = device)

trainloader = DataLoader(TensorDataset(data, target_),
                         batch_size=50,
                         shuffle=True)

model = LSTMEEG(170, 30)
his = train_LSTMEEG(model, trainloader, 100, 0.01)
'''