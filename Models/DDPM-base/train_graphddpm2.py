# -*- coding: utf-8 -*-
import os
import torch
import argparse
import numpy as np
import torch.utils.data
from easydict import EasyDict as edict
from timeit import default_timer as timer

from utils.eval import Metric
from utils.gpu_dispatch import GPU
from utils.common_utils import dir_check, to_device, ws, unfold_dict, dict_merge, GpuId2CudaId, Logger

from algorithm.dataset import CleanDataset, TrafficDataset
from algorithm.diffgraph2.model import DiffSTG, save2file
'''
try:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir = os.path.join(os.environ["NNI_OUTPUT_DIR"], 'tensorboard'))
except:
    pass
'''
def setup_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_params():
    parser = argparse.ArgumentParser(description='Entry point of the code')

    # Model parameters
    # parser.add_argument("--epsilon_theta", type=str, default='GSTNet')
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--N", type=int, default=200) # noising step
    parser.add_argument("--beta_schedule", type=str, default='quad')
    parser.add_argument("--beta_end", type=float, default=0.1)
    parser.add_argument("--sample_steps", type=int, default=200) # ddim
    parser.add_argument("--ss", type=str, default='ddpm')
    parser.add_argument("--T_h", type=int, default=12)

    # Evaluation parameters
    parser.add_argument('--n_samples', type=int, default=8)

    # Training parameters
    parser.add_argument("--is_train", type=bool, default=True)
    parser.add_argument("--data", type=str, default='PEMS08')
    # parser.add_argument("--mask_ratio", type=float, default=0.0)
    parser.add_argument("--is_test", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--batch_size", type=int, default = 64)
    parser.add_argument("--epoch", type=int, default = 200)
    parser.add_argument('--num_layers', type= int, default=3)

    args, _ = parser.parse_known_args()
    return args

def default_config(data='AIR_BJ',args = None):
    config = edict()
    config.PATH_MOD = ws + '/output/model/'
    config.PATH_LOG = ws + '/output/log/'
    config.PATH_FORECAST = ws + '/output/forecast/'

    # Data Config
    config.data = edict()
    config.data.name = data
    config.data.path = ws + '/data/dataset/'

    config.data.feature_file = config.data.path + config.data.name + '/flow.npy'  # Add this line
    config.data.spatial = config.data.path + config.data.name + '/adj.npy'
    config.data.num_recent = 1

    # Data settings for different datasets
    if config.data.name == 'PEMS08':
        config.data.num_vertices = 170
        config.data.points_per_hour = 12
        config.data.val_start_idx = int(17856 * 0.6)
        config.data.test_start_idx = int(17856 * 0.8)

    if config.data.name == "AIR_BJ":
        config.data.num_vertices = 34
        config.data.points_per_hour = 1
        config.data.val_start_idx = int(8760 * 0.6)
        config.data.test_start_idx = int(8760 * 0.8)

    if config.data.name == 'AIR_GZ':
        config.data.num_vertices = 41
        config.data.points_per_hour = 1
        config.data.val_start_idx = int(8760 * 10 / 12)
        config.data.test_start_idx = int(8160 * 11 / 12)

    gpu_id = GPU().get_usefuel_gpu(max_memory=60000, condidate_gpu_id=[0])
    config.gpu_id = gpu_id
    print(gpu_id)
    if gpu_id != None:
        cuda_id = GpuId2CudaId(gpu_id)
        print(cuda_id)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device("cuda:0")
    print(device)

    # device = torch.device('mps')
    # print(device)

    # Model config
    config.model = edict()
    config.model.V = config.data.num_vertices
    config.model.F = 1
    config.model.rnn_num_unit = args.hidden_size
    # config.model.week_len = 7

    # data
    config.model.day_len = config.data.points_per_hour * 24
    config.model.T_p = 12
    config.model.T_h = 12
    config.model.device = device
    # config.model.d_h = 32
    config.device = config.model.device

    # Diffusion model config
    config.model.N = args.N
    config.model.sample_steps = 200
    config.model.epsilon_theta = 'GSTNet'
    # config.model.is_label_condition = True
    config.model.beta_end = args.beta_end
    config.model.beta_schedule = args.beta_schedule
    config.model.sample_strategy = args.ss
    config.model.graph_diffusion_step = 3
    config.model.cheb_k = 3
    config.model.num_layers = args.num_layers

    config.n_samples = args.n_samples

    # Training config
    config.model_name = 'DiffSTG'
    config.is_test = False
    config.epoch = args.epoch
    config.optimizer = "adam"
    config.lr = args.lr
    config.batch_size = args.batch_size
    # config.mask_ratio = 0.0

    # config.wd = 1e-5
    config.early_stop = 10
    config.start_epoch = 0
    # config.device = device
    config.logger = Logger()

    if not os.path.exists(config.PATH_MOD):
        os.makedirs(config.PATH_MOD)
    if not os.path.exists(config.PATH_LOG):
        os.makedirs(config.PATH_LOG)
    if not os.path.exists(config.PATH_FORECAST):
        os.makedirs(config.PATH_FORECAST)
    return config

def evals(model, data_loader, epoch, metric, config, clean_data, mode='Test'):
    setup_seed(2022)
    print('start eval process')

    y_pred, y_true, time_lst = [], [], []
    metrics_future = Metric(T_p=config.model.T_p)
    metrics_history = Metric(T_p=config.model.T_h)
    model.eval()

    samples, targets = [], []
    for i, batch in enumerate(data_loader):
        if i > 0 and config.is_test:
            break
        time_start = timer()

        future, history, pos_w, pos_d = to_device(batch, config.device)
        targets.append(future.cpu())

        n_samples = 1 if mode == 'Val' else config.n_samples
        x_hat = model(future, history, n_samples)
        samples.append(x_hat.transpose(2,4).cpu())

        if x_hat.shape[-1] != (config.model.T_h + config.model.T_p):
            x_hat = x_hat.transpose(2,4)

        time_lst.append((timer() - time_start))
        future, x_hat = clean_data.reverse_normalization(future), clean_data.reverse_normalization(x_hat)
        x_hat = x_hat.detach()
        f_x, f_x_hat = future, x_hat

        _y_true_ = f_x.transpose(1, 3).cpu().numpy()
        _y_pred_ = f_x_hat.transpose(2, 4).cpu().numpy()
        _y_pred_ = np.clip(_y_pred_, 0, np.inf)
        metrics_future.update_metrics(_y_true_, _y_pred_)

        y_pred.append(_y_pred_)
        y_true.append(_y_true_)
        '''
        h_x, h_x_hat = x[:, :, :, :config.model.T_h], x_hat[:, :, :, :,  :config.model.T_h]
        _y_true_ = h_x.transpose(1, 3).cpu().numpy()
        _y_pred_ = h_x_hat.transpose(2, 4).cpu().numpy()
        _y_pred_ = np.clip(_y_pred_, 0, np.inf)
        metrics_history.update_metrics(_y_true_, _y_pred_)
        '''
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    time_cost = np.sum(time_lst)
    metric.update_metrics(y_true, y_pred)
    metric.update_best_metrics(epoch=epoch)
    metric.metrics['time'] = time_cost

    if mode == 'test':
        samples = torch.cat(samples, dim=0)[:50]
        targets = torch.cat(targets, dim=0)[:50]
        observed_flag = torch.ones_like(targets)
        evaluate_flag = observed_flag
        evaluate_flag[:, -config.model.T_p:, :, :] = 1
        import pickle
        with open(config.forecast_path, 'wb') as f:
            pickle.dump([samples, targets, observed_flag, evaluate_flag], f)

        message = f"predict_path = '{config.forecast_path}'"
        config.logger.message_buffer += f"{message}\n"
        config.logger.write_message_buffer()

    if metric.best_metrics['epoch'] == epoch:
        message = f" |[{metric.metrics['mae']:<7.2f}{metric.metrics['rmse']:<7.2f}]"
    else:
        message = f" | {metric.metrics['mae']:<7.2f}{metric.metrics['rmse']:<7.2f}"
    print(message, end='', flush=False)
    config.logger.message_buffer += message

    message = f" | {metrics_history.metrics['mae']:<7.2f}{metrics_history.metrics['rmse']:<7.2f}{time_cost:<5.2f}s"
    print(message, end='\n', flush=False)
    config.logger.message_buffer += f"{message}\n"

    config.logger.write_message_buffer()
    torch.cuda.empty_cache()
    return metric

def main():
    # torch.set_num_threads(2)
    args = get_params()
    setup_seed(0)
    config = default_config("PEMS08", args)
    params = unfold_dict(config)

    config.T_h = config.model.T_h
    config.T_p = config.model.T_p

    if config.model.sample_steps > config.model.N:
        print('sample steps large than N, exit')
        return 0

    config.trial_name = '+'.join([f"{v}" for k, v in params.items()])
    config.log_path = f"{config.PATH_LOG}/{config.trial_name}.log"

    dir_check(config.log_path)
    config.logger.open(config.log_path, mode="w")
    config.logger.write(config.__str__() + '\n', is_terminal=False)

    clean_data = CleanDataset(config)

    print("config.device:", config.device)
    model = DiffSTG(config.model)

    model = model.to(config.model.device)

    train_dataset = TrafficDataset(clean_data, (0 + config.model.T_p, config.data.val_start_idx - config.model.T_p + 1), config)
    train_loader = torch.utils.data.DataLoader(train_dataset, config.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    val_dataset = TrafficDataset(clean_data, (config.data.val_start_idx + config.model.T_p, config.data.test_start_idx - config.model.T_p + 1), config)
    val_loader = torch.utils.data.DataLoader(val_dataset, config.batch_size, shuffle=False)

    test_dataset = TrafficDataset(clean_data, (config.data.test_start_idx + config.model.T_p, -1), config)
    test_loader = torch.utils.data.DataLoader(test_dataset, config.batch_size, shuffle=False)

    print('learning rate:', config.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    metrics_val = Metric(T_p=config.model.T_h + config.model.T_p)

    model_path = config.PATH_MOD + config.trial_name
    config.model_path = model_path
    config.logger.write(f"model_path = '{model_path}'\n", is_terminal=True)

    config.forecast_path = forecast_path = config.PATH_FORECAST + config.trial_name + '.pkl'
    config.logger.write(f"forecast_path:{model_path}\n", is_terminal=False)
    print('forecast_path:', forecast_path)
    dir_check(forecast_path)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"total parameters of the model: {total_params}")

    total_params = sum(p.numel() for p in model.eps_model.parameters())
    print(f"total parameters of the eps_model: {total_params}")

    best_epoch = 0
    train_start_t = timer()
    # Train and sample the data
    print(config.model.device)
    print(model.device)
    torch.cuda.empty_cache()
    for epoch in range(config.epoch):
        # if not params['is_train']: break
        if epoch > 1 and config.is_test: 
            print("config.is_test:", config.is_test)
            print(epoch > 1 and config.is_test)
            break

        n, avg_loss, time_lst = 0, 0, []
        # train diffusion model
        batches_seen = 0
        for i, batch in enumerate(train_loader):
            if i > 3 and config.is_test:break
            time_start =  timer()
            future, history, pos_w, pos_d = batch # future:(B, T_p, V, F), history: (B, T_h, V, F)

            # get x0
            history = history.to(config.device)
            future = history.to(config.device)

            # loss calculate
            eps = torch.randn_like(future)

            loss = 100 * model.denoising(future, history, eps, batches_seen)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate the moving average training loss
            batches_seen += 1
            n += 1
            avg_loss = avg_loss * (n - 1) / n + loss.item() / n
            
            time_lst.append((timer() - time_start))
            message = f"{i / len(train_loader) + epoch:6.1f}| {avg_loss:0.3f} {np.sum(time_lst):.1f}s"
            print('\r' + message, end='', flush=True)

        config.logger.message_buffer += message
        '''
        try:
            writer.add_scalar('train/loss', avg_loss, epoch)
        except:
            pass
        '''
    metric_lst = []
    for sample_strategy, sample_steps in [('ddpm', 200)]:
        if sample_steps > config.model.N: break

        config.model.sample_strategy = sample_strategy
        config.model.sample_steps = sample_steps

        model.set_ddim_sample_steps(sample_steps)
        model.set_sample_strategy(sample_strategy)

        metrics_test = Metric(T_p=config.model.T_h + config.model.T_p)
        evals(model, test_loader, epoch, metrics_test, config, clean_data, mode='test')
        message = f'sample_strategy:{sample_strategy}, sample_steps:{sample_steps} Final results in test:{metrics_test}\n'
        #print('------------------------')
        #message = f'sample_strategy:{sample_strategy}, sample_steps:{sample_steps} Final results in test: mse:{metrics_test[0]:.2f}, mae:{metrics_test[1]:.2f}, rmse:{metrics_test[2]:.2f}, r2_score:{metrics_test[3]:.2f}, mape:{metrics_test[4]:.2f} | {metrics_test[5]}\n'

        config.logger.write(message, is_terminal=True)

        params = unfold_dict(config)
        params = dict_merge([params, metrics_test.to_dict()])
        params['best_epoch'] = metrics_val.best_metrics['epoch']
        params['model'] = config.model.epsilon_theta
        save2file(params)
        metric_lst.append(metrics_test.metrics['mae'])

    # rename log file
    log_file, log_name = os.path.split(config.log_path)
    new_log_path = os.path.join(log_file, f"[{config.data.name}]mae{min(metric_lst):7.2f}+{log_name}")

if __name__ == "__main__":
    main()