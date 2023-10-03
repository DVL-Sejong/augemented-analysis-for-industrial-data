import pandas as pd
import numpy as np
import torch

from time import time
from utils_single import Dataset, make_rbf_df
from models import MultiRBFnnTime,MultiRBFnn_sigma

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type = int, default = 3000)
parser.add_argument('--lr', type = float, default=1e-3)
parser.add_argument('--add_rbf_num', type=int, default=100)
parser.add_argument('--humanth', type = int, default=0)
parser.add_argument('--missing_rate', type= int, default=10)
parser.add_argument('--sigma', type= str, default='time')

args = parser.parse_args()


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    human = ["A", 'B', 'C', 'D', 'E']
    num = ["01", "02", "03", "04", "05"]
    Tag = ['010-000-024-033', '010-000-030-096', '020-000-033-111', '020-000-032-221']
    if args.sigma == "time":
        for humanth in range(5):
            for numth in range(5):
                for Tagth in range(4):

                    save_name = "{}_rbf_data{}_{}_time.csv".format(human[humanth]+num[numth], args.missing_rate, Tag[Tagth])
                    save_path = "./singleRBFresult/Timeresult/missing{0}/".format(args.missing_rate)
                    save_model = "{}_rbf_{}_{}.pt".format(human[args.humanth]+num[numth], args.missing_rate, Tag[Tagth])

                    input_data, target, lossth, target_ground, missing_index = Dataset(humanth, numth, Tagth, args.missing_rate, device)
                    print("loss th:", lossth)
                    pred_list = []
                    for i in range(target.size(0)):
                        print("{}th train start".format(i))
                        model = MultiRBFnnTime(1, add_rbf_num = args.add_rbf_num, device=device)
                        model.train(input_data = input_data, target = target[i], epochs = 3000,
                                        lr = args.lr, loss_th=lossth, lr_change_th=lossth)
                        
                        model_pred = model.pred(input_data)[1]
                        pred_list.append(model_pred)
                        print(torch.mean(torch.abs(target_ground[i][(missing_index[i] != 1)] -  model_pred[(missing_index[i] != 1)])))
                        print(torch.mean(torch.abs(target_ground[i][(missing_index[i] != 0)] -  model_pred[(missing_index[i] != 0)])))

                    rbf_df = make_rbf_df(humanth, numth, Tagth, args.missing_rate, pred_list)

                    rbf_df.to_csv(save_path + save_name, index = False)
    else:
        print("sigma : 1 train")
        for humanth in range(5):
            for numth in range(5):
                for Tagth in range(4):
                    save_name = "{}_rbf_data{}_{}_random.csv".format(human[humanth]+num[numth], args.missing_rate, Tag[Tagth])
                    save_path = "./singleRBFresult/randomresult/missing{0}/".format(args.missing_rate)
                    save_model = "{}_rbf_{}_{}.pt".format(human[args.humanth]+num[numth], args.missing_rate, Tag[Tagth])

                    input_data, target, lossth, target_ground, missing_index = Dataset(humanth, numth, Tagth, args.missing_rate, device)
                    print("loss th:", lossth)
                    pred_list = []
                    for i in range(target.size(0)):
                        print("{} {}th train start".format(human[humanth],i))
                        model = MultiRBFnn_sigma(1, add_rbf_num = args.add_rbf_num, device=device)
                        model.train(input_data = input_data, target = target[i], epochs = 3000,
                                        lr = args.lr, loss_th=lossth, lr_change_th=lossth)
                        
                        model_pred = model.pred(input_data)[1]
                        pred_list.append(model_pred)

                        print(torch.mean(torch.abs(target_ground[i][(missing_index[i] != 1)] -  model_pred[(missing_index[i] != 1)])))
                        print(torch.mean(torch.abs(target_ground[i][(missing_index[i] != 0)] -  model_pred[(missing_index[i] != 0)])))

                    rbf_df = make_rbf_df(humanth, numth, Tagth, args.missing_rate, pred_list)

                    rbf_df.to_csv(save_path + save_name, index = False)

if __name__ == '__main__':
    run()
    






