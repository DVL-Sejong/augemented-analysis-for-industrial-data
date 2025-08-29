import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
device = "cuda" if torch.cuda.is_available() else "cpu"

def make_deltas2(masks):
    deltas = []
    for h in range(len(masks)):
        if h == 0:
            deltas.append([1 for _ in range(masks.shape[1])])
        else:
            deltas.append([1 for _ in range(masks.shape[1])] + (1-masks[h]) * deltas[-1])
    
    return list(deltas)


class MyDataset2(Dataset):
    def __init__(self, dataset, q):
        self.data = dataset
        self.q = q

    def __len__(self):
        return self.data.shape[1] // self.q

    def __getitem__(self, index):
        return self.data[:,index * self.q : index * self.q + self.q,:]

def missing_data_rbf2(df,rbf, batch_size):
    
    values = ((df - df.mean()) / df.std()).values
    shp = values.shape
    rbf_df = pd.read_csv("./RBFresult/" + rbf)
    masks = ~np.isnan(values)
    
    masks = masks.reshape(shp)

    deltas = np.array(make_deltas2(masks))
    values = torch.nan_to_num(torch.from_numpy(values).to(torch.float32))
    masks = torch.from_numpy(masks).to(torch.float32)
    deltas = torch.from_numpy(deltas).to(torch.float32)
    rbf_x = torch.from_numpy(rbf_df.values).to(torch.float32)
    dataset = torch.cat([values.unsqueeze_(0), deltas.unsqueeze_(0), masks.unsqueeze_(0), rbf_x.unsqueeze_(0)], dim = 0)
    
    mydata  = MyDataset2(dataset, batch_size)
    data = DataLoader(mydata, batch_size, shuffle=True)

    return data

def val_missing_data_rbf2(df,rbf):
    
    values = ((df - df.mean()) / df.std()).values
    shp = values.shape
    rbf_df = pd.read_csv("./RBFresult/" + rbf)
    
    masks = ~np.isnan(values)
    
    masks = masks.reshape(shp)

    deltas = np.array(make_deltas2(masks))
    values = torch.nan_to_num(torch.from_numpy(values).to(torch.float32))
    masks = torch.from_numpy(masks).to(torch.float32)
    deltas = torch.from_numpy(deltas).to(torch.float32)
    rbf_x = torch.from_numpy(rbf_df.values).to(torch.float32)
    dataset = torch.cat([values.unsqueeze_(0), deltas.unsqueeze_(0), masks.unsqueeze_(0), rbf_x.unsqueeze_(0)], dim = 0).unsqueeze_(0)

    return dataset

def eval_model2(model, rbf, realpath, dfpath):
    
    df = pd.read_csv("./dataset/" + dfpath).drop(['datetime'], axis = 1)
    dataset = val_missing_data_rbf2(df,rbf)
    dataset = dataset.to(device)

    real = pd.read_csv("./dataset/" + realpath).drop(['datetime'], axis = 1)
    real_scaler = (real - df.mean()) / df.std()

    df_scaler = ((df-df.mean()) / df.std()).values
    masks = ~np.isnan(df_scaler)
    masks = torch.from_numpy(masks).to(torch.float32)
    
    eval_masks = ~np.isnan(real_scaler.values)
    eval_masks = torch.from_numpy(eval_masks).to(torch.float32)

    test_masks = eval_masks - masks
    real_scaler = torch.nan_to_num(torch.from_numpy(real_scaler.values).to(torch.float32))
    
    model.eval()
    imputations, x_loss, c_hat_list = model(dataset)

    Nonscale_imputataion = pd.DataFrame(imputations[0].cpu().detach() , columns= df.columns)
    Nonscale_imputataion = (Nonscale_imputataion * df.std()) + df.mean()
    
    real = real.fillna(0)
    print("Scale MAE :", torch.sum(torch.abs(imputations[0].cpu().detach() - real_scaler) * test_masks) / torch.sum(test_masks))
    print("Scale MRE :", torch.sum(torch.abs(imputations[0].cpu().detach() - real_scaler) * test_masks) / torch.sum(torch.abs(real_scaler * test_masks)))

    print("Original MAE :", np.sum(np.abs((Nonscale_imputataion - real).values * test_masks.cpu().numpy())) / np.sum(test_masks.cpu().numpy()))
    print("Original MRE :", np.sum(np.abs((Nonscale_imputataion - real).values * test_masks.cpu().numpy())) / np.sum(np.abs(real.values * test_masks.cpu().numpy())))

    return Nonscale_imputataion

def eval_bi_model(model, rbf, realpath, dfpath):
    
    df = pd.read_csv("./dataset/" + dfpath).drop(['datetime'], axis = 1)
    dataset = val_missing_data_rbf2(df,rbf)
    dataset = dataset.to(device)

    real = pd.read_csv("./dataset/" + realpath).drop(['datetime'], axis = 1)
    real_scaler = (real - df.mean()) / df.std()

    df_scaler = ((df-df.mean()) / df.std()).values
    masks = ~np.isnan(df_scaler)
    masks = torch.from_numpy(masks).to(torch.float32)
    
    eval_masks = ~np.isnan(real_scaler.values)
    eval_masks = torch.from_numpy(eval_masks).to(torch.float32)

    test_masks = eval_masks - masks
    real_scaler = torch.nan_to_num(torch.from_numpy(real_scaler.values).to(torch.float32))
    
    model.eval()
    loss, x_loss, back_x_loss, loss_c, biimputataion = model(dataset)

    Nonscale_imputataion = pd.DataFrame(biimputataion[0].cpu().detach() , columns= df.columns)
    Nonscale_imputataion = (Nonscale_imputataion * df.std()) + df.mean()
    
    real = real.fillna(0)
    print("Scale MAE :", torch.sum(torch.abs(biimputataion[0].cpu().detach() - real_scaler) * test_masks) / torch.sum(test_masks))
    print("Scale MRE :", torch.sum(torch.abs(biimputataion[0].cpu().detach() - real_scaler) * test_masks) / torch.sum(torch.abs(real_scaler * test_masks)))

    print("Original MAE :", np.sum(np.abs((Nonscale_imputataion - real).values * test_masks.cpu().numpy())) / np.sum(test_masks.cpu().numpy()))
    print("Original MRE :", np.sum(np.abs((Nonscale_imputataion - real).values * test_masks.cpu().numpy())) / np.sum(np.abs(real.values * test_masks.cpu().numpy())))

    return Nonscale_imputataion