import os
import torch
import yaml
import torchmetrics

from argparse import Namespace
from tasks.supervised import SupervisedForecastTask
from models.tgcn import TGCN
from models.gcn import GCN
from utils.data.spatiotemporal_csv_data import SpatioTemporalCSVDataModule

def get_args(main_path, ver_path, model_file, data_type, missing_ratio, data):
    args = Namespace(
        model_path = main_path + ver_path + '/checkpoints/' + model_file,
        feat_path = "data/" + data_type + "/speed_" + data_type + "_" + missing_ratio + ".csv",
        adj_path = "data/" + data_type + "/adj_mx_" + data_type + ".csv",
        batch_size = data['batch_size'],
        seq_len = data['seq_len'],
        pre_len = data['pre_len'],
        split_ratio = data['split_ratio'],
        hidden_dim = data['hidden_dim'],
        learning_rate = data['learning_rate'],
        gpus = data['gpus'],
        weight_decay = data['weight_decay'],
        loss = data['loss']
    )
    return args


def get_DataModule(args, data_type):
    dm = SpatioTemporalCSVDataModule(
        feat_path=args.feat_path, adj_path=args.adj_path, pre_len=args.pre_len
    )

    dm_eval = SpatioTemporalCSVDataModule(
        feat_path="data/" + data_type + "/speed_" + data_type + "_0.csv", adj_path=args.adj_path, pre_len=args.pre_len
    )

    return dm, dm_eval


def get_model_and_task(data, dm, args):
    if data['model_name'] == 'GCN':
        model = GCN(adj=dm.adj, input_dim=args.seq_len, output_dim=args.hidden_dim)
    else:
        model = TGCN(adj=dm.adj, hidden_dim=args.hidden_dim)

    task = SupervisedForecastTask(model=model, feat_max_val=dm.feat_max_val, **vars(args))
    task = task.load_from_checkpoint(args.model_path)

    return model, task


def data_loading(dm, dm_eval):
    for batch in dm.val_dataloader():
        x, _ = batch
        pred = task(x)

    for val in dm_eval.val_dataloader():
        pass
    y = val[1]

    num_nodes = x.size(2)
    y = torch.where(y == 0, 0.01, y)

    pred = pred.transpose(1, 2).reshape((-1, num_nodes))
    y = y.reshape((-1, y.size(2)))

    pred = pred * task.feat_max_val
    y = y * task.feat_max_val

    return pred, y


def evaluation(pred, y):
    mean_abs_percentage_error = torchmetrics.MeanAbsolutePercentageError()

    mae = torchmetrics.functional.mean_absolute_error(pred, y)
    rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(pred, y))
    mape = mean_abs_percentage_error(pred, y)

    return mae.detach().numpy(), rmse.detach().numpy(), mape.detach().numpy()

if __name__ == '__main__':
    main_path = "lightning_logs/"
    for ver_path in os.listdir(main_path):
        model_file = os.listdir(main_path + ver_path + "/checkpoints/")[0]

        with open(main_path + ver_path + "/hparams.yaml", "r") as file:
            data = yaml.load(file, Loader=yaml.Loader)
        data_file = data['data']
        data_type = data_file.split('_')[0]
        missing_ratio = data_file.split('_')[1]
        if data_type == 'gang':
            data_type = 'gangnam'

        args = get_args(main_path, ver_path, model_file, data_type, missing_ratio, data)
        dm, dm_eval = get_DataModule(args, data_type)

        dm.setup()
        dm_eval.setup()

        model, task = get_model_and_task(data, dm, args)
        task.eval()

        pred, y = data_loading(dm, dm_eval)

        mae, rmse, mape = evaluation(pred, y)

        print("Log Path:", main_path + ver_path)
        print("Model:", data['model_name'])
        print("Data:", data_type, ", Missing:", missing_ratio, "%, Prediction:", args.pre_len * 5, 'min')
        print("MAE:", mae, ", RMSE:", rmse, ", MAPE:", mape)

        print('----------------------------')