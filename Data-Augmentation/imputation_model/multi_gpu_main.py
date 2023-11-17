import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from models import BiMGRU_ori, train_BiMGRU
from utils import missing_data_ori, eval_model_ori
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type = int, default = 1000)
parser.add_argument('--lr', type = float, default=1e-3)

parser.add_argument('--input_size', type = int, default = 3480)
parser.add_argument('--hidden_size', type= int, default = 64)
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--seqlen', type = int, default = 40)
parser.add_argument('--models', type = str, default = 'brits')
args = parser.parse_args()

def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = './result/'
    model_name = 'G_{}_{}_{}_{}'.format(args.lr, args.hidden_size, args.seqlen, args.models)
    print('device:',device)

    if args.models == 'brits':
        print('BRITS train')
        dataset = missing_data_ori(args.batch_size, args.seqlen)
        G = BiMGRU_ori(args.input_size, args.hidden_size)
        train_BiMGRU(G, args.lr, args.epochs, dataset, device)
        torch.save(G, save_path + model_name + '.pt')

        print('eval_start')
        Nonscale_imputataion = eval_model_ori(G)

        Nonscale_imputataion.to_csv('./result/imputation_traffic_epochs{}.csv'.format(args.epochs), index = False)
        print('done')
if __name__ == '__main__':
    run()