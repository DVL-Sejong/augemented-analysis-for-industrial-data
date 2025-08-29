from torch.utils.data import Dataset, DataLoader

import torch
import numpy as np
from torch.nn.parallel import DataParallel
import pandas as pd
device = "cuda" if torch.cuda.is_available() else "cpu"

def make_deltas(masks):
    deltas = []
    for h in range(len(masks)):
        if h == 0:
            deltas.append([1 for _ in range(masks.shape[1])])
        else:
            deltas.append([1 for _ in range(masks.shape[1])] + (1-masks[h]) * deltas[-1])
    
    return list(deltas)

class MyDataset(Dataset):
    def __init__(self, dataset, q):
        self.data = dataset
        self.q = q

    def __len__(self):
        return self.data.shape[1] // self.q

    def __getitem__(self, index):
        return self.data[:,index * self.q : index * self.q + self.q,:]
    
def missing_data_ori(batch_size, seq_len):
    df = pd.read_csv("./dataset/speed_all_node_missing.csv")
    
    values = ((df - df.mean()) / df.std()).values
    shp = values.shape
    masks = ~np.isnan(values)
    
    masks = masks.reshape(shp)

    deltas = np.array(make_deltas(masks))
    values = torch.nan_to_num(torch.from_numpy(values).to(torch.float32))
    masks = torch.from_numpy(masks).to(torch.float32)
    deltas = torch.from_numpy(deltas).to(torch.float32)
    dataset = torch.cat([values.unsqueeze_(0), deltas.unsqueeze_(0), masks.unsqueeze_(0)], dim = 0)

    mydata  = MyDataset(dataset, seq_len)
    data = DataLoader(mydata, batch_size, shuffle=False, num_workers=4)

    return data



def eval_model_ori(model):
    
    df = pd.read_csv("./dataset/speed_all_node_missing.csv")
    values = ((df - df.mean()) / df.std()).values
    shp = values.shape

    masks = ~np.isnan(values)
    masks = masks.reshape(shp)
    eval_masks = np.ones_like(masks) - masks.astype(float)

    deltas = np.array(make_deltas(masks))
    values = torch.nan_to_num(torch.from_numpy(values).to(torch.float32))
    masks = torch.from_numpy(masks).to(torch.float32)
    deltas = torch.from_numpy(deltas).to(torch.float32)

    dataset = torch.cat([values.unsqueeze_(0), deltas.unsqueeze_(0), masks.unsqueeze_(0)], dim = 0).unsqueeze_(0)

    model = DataParallel(model, output_device=0)
    model.eval()
    with torch.no_grad():
        dataset = dataset.to(device)
        loss, x_loss, back_x_loss, loss_c, bi_chat, biimputataion = model(dataset)

    Nonscale_imputataion = pd.DataFrame(biimputataion[0].cpu().detach() , columns= df.columns)
    Nonscale_imputataion = (Nonscale_imputataion * df.std()) + df.mean()

    return Nonscale_imputataion
