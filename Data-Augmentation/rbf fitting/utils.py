from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
import torch
import argparse
import torch.nn as nn

def data_load(missing_df, ground_df, path):
    df = pd.read_csv(
    path + str(missing_df), sep=",",header=0)
    df_ground = pd.read_csv(
        path + str(ground_df)', sep=",",header=0)

    df = df.drop(['datetime'], axis = 1)
    df_ground = df_ground.drop(['datetime'], axis = 1)
    
    return df, df_ground

def make_deltas(masks):
    deltas = []
    for h in range(len(masks)):
        if h == 0:
            deltas.append([1 for _ in range(masks.shape[1])])
        else:
            deltas.append([1 for _ in range(masks.shape[1])] + (1-masks[h]) * deltas[-1])
    
    return list(deltas)

def missing_data(df, ground_df):

    evals = []

    # for h in range(len(df)):
    #     evals.append((data.iloc[h]))
    
    values = ((df - ground_df.mean()) / ground_df.std()).values
    evals = ((ground_df - ground_df.mean())/ ground_df.std()).values
    shp = evals.shape

    # values = evals.copy()
    
    masks = ~np.isnan(values)
    eval_masks = masks.copy()
    
    masks = masks.reshape(shp)
    eval_masks = eval_masks.reshape(shp)

    deltas = np.array(make_deltas(masks))
    
    values = torch.nan_to_num(torch.from_numpy(values).unsqueeze(0).to(torch.float32))
    masks = torch.from_numpy(masks).unsqueeze(0).to(torch.float32)
    evals = torch.from_numpy(evals).unsqueeze(0).to(torch.float32)
    eval_masks = torch.from_numpy(eval_masks).unsqueeze(0).to(torch.float32)
    deltas = torch.from_numpy(deltas).unsqueeze(0).to(torch.float32)
    

    return values, masks, evals, eval_masks, deltas 


class MyDataset(Dataset):
    def __init__(self, df : pd.DataFrame, ground_df : pd.DataFrame):
        self.datas, self.masks, self.evals, self.eval_masks, self.deltas = missing_data(df,ground_df)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        # 여기 코드를 고치면 sequence 를 조정할 수 있음
        return self.datas[idx], self.masks[idx], self.evals[idx], self.eval_masks[idx], self.deltas[idx]
    
def get_loader(df, ground_df, batch_size = 256, shuffle = False):
    data_set = MyDataset(df,ground_df)
    data_iter = DataLoader(dataset = data_set, batch_size = batch_size,num_workers = 1, shuffle = shuffle, pin_memory = True)

    return data_iter