import time
import utils
import models
import argparse
import pandas as pd
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import pickle

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type = int, default = 1000)
parser.add_argument('--batch_size', type = int, default = 32)
parser.add_argument('--model', type = str)
parser.add_argument('--input_size', type = int, default = 36)
parser.add_argument('--hidden_size',type = int, default = 108)
parser.add_argumnet('--path', type = str, default = './dataset/')
parser.add_argumnet('--df')
parser.add_argument('--ground_df')


args = parser.parse_args()


def train(model):
    
    df, ground_df = utils.data_load(missing_df = args.df, ground_df = args.ground_df, path = args.path)
    data_iter = utils.get_loader(df, ground_df, batch_size = args.batch_size)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    
    run_loss = 0.0
    model.train()
    progress = tqdm(range(args.epochs))
    
    imputation_list = []
    loss_list = []
    for epoch in progress:
        batch_loss = 0.0
        imputations = []
        for values, masks, evals, eval_masks, deltas in data_iter:
            values = values.to(device)
            masks = masks.to(device)
            deltas = deltas.to(device)
            x_loss, x_imputation = model(values, masks, deltas)
            batch_loss += x_loss

        optimizer.zero_grad()
        x_loss.backward()
        optimizer.step()
        imputation_list.append(x_imputation.detach().cpu().numpy())
        progress.set_description("loss: {:0.3f}".format(x_loss/len(data_iter)))
        loss_list.append(x_loss.detach().cpu().numpy())
    
    with open('loss.pickle', 'w') as f:
        f.write(x_loss)
       
    with open('imputation.pickle', 'w') as f:
        f.write(imputation_list)
        
    return loss_list, imputation_list


def run():
    model = getattr(models, args.model).Model(input_size = args.input_size, hidden_size = args.hidden_size)
    
    if torch.cuda.is_available():
        model = model.cuda()
        
        train(model)
        
if __name__ == '__main__':
    run()        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    