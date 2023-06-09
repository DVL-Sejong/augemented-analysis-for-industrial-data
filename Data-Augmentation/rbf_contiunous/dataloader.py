import pandas as pd
import numpy as np
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

def Dataset(dfpath, df_ground_path):
    df = pd.read_csv('./dataset/'+ dfpath, sep=",",header=0)
    df_ground = pd.read_csv('./dataset/'+ df_ground_path, sep=",",header=0)
    df = df.drop(['datetime'], axis = 1)
    df_ground = df_ground.drop(['datetime'], axis = 1)

    input_data =  torch.tensor(np.array(df.index) + 1, device = device, dtype = torch.float64)
    target = torch.tensor(df.values.T.astype(np.float64), device = device, dtype = torch.float64)
    target_ground = torch.tensor(df_ground.values.T.astype(np.float64), device = device, dtype = torch.float64)

    return input_data, target, target_ground

def gaussian(input_data, centers, sigma, weights):
    out = torch.exp(-1 *(torch.pow((input_data - centers), 2)) / (torch.pow(sigma, 2)))
    pred = torch.mm(weights, out)
    return pred

def generation_dataset():
    
    r_c = torch.tensor([1.32141,7.123, 16.1256142, 24.21512, 33.16512,52.13, 47.1626, 58.735, 72.1624], dtype = torch.float64 ,device= device).reshape(9,1)
    r_s = torch.tensor([1.1,2.3, 3.6142, 7.21512, 11.162, 3.626, 2.735, 6.1624, 1.2], dtype = torch.float64 ,device= device).reshape(9,1)
    r_w = torch.tensor([[-10,4, 12, 32, -33, 12, -20, 3, 10],
                        [12,3, -1, -22, 3, 32, -20, 4, 2],
                        [-10,5, 3, 23, -13, 23, 17, 2, 5]], dtype = torch.float64, device = device)
    
    input_ = np.arange(1,80,1)
    input_ = torch.tensor(input_, dtype = torch.float64,device = device)
    target = gaussian(input_, r_c, r_s, r_w)
    target2 = gaussian(input_, r_c, r_s, r_w)

    for i in range(input_.size()[0] // 3):
        target[0][(i+1) * 3] = torch.nan
    for j in range(input_.size()[0] // 4):
        target[1][(j+1) * 4] = torch.nan
    for k in range(input_.size()[0] // 5):
        target[2][(k+1) * 5] = torch.nan

    return input_, target, target2

    
