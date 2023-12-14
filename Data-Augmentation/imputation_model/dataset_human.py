import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch

class Human_Dataset(Dataset):
    def __init__(self, eval_length=36, target_dim=3, use_index_list=None, missing_ratio=0.0):
        self.eval_length = eval_length
        self.target_dim = target_dim

        df_full = pd.read_csv("./data/missing{}/{}_missing.csv".format(missing_ratio, missing_ratio))
        df_full = df_full[['x_coordinate', 'y_coordinate', 'z_coordinate']]

        # create data for batch
        self.observed_data = []  # values (separated into each month)
        self.gt_data = []  # masks (separated into each month)
        self.observed_masks = []
        self.gt_mask = []  # ground-truth masks (separated into each month)
        self.index_month = []  # indicate month
        self.position_in_month = []  # indicate the start position in month (length is the same as index_month)
        self.valid_for_histmask = []  # whether the sample is used for histmask
        self.use_index = []  # to separate train/valid/test
        self.cut_length = []  # excluded from evaluation targets
        
        for human in ["A", 'B', 'C', 'D', 'E']:
            for num in ["01", "02", "03", "04", "05"]:
                for Tag in ['010-000-024-033', '010-000-030-096', '020-000-033-111', '020-000-032-221']:
                    current_df = pd.read_csv('./data/missing{0}/Tagdataset{0}/{1}_data{0}_{2}.csv'.format(missing_ratio, human + num, Tag))
                    current_df = current_df[['x_coordinate', 'y_coordinate', 'z_coordinate']]
                    # print(c_data.shape)

                    real = pd.read_csv('./data/Tagdataset/{}_data_{}.csv'.format(human + num, Tag))[['x_coordinate', 'y_coordinate', 'z_coordinate']]
                    real = ((real - df_full.mean()) /  df_full.std()).values

                    c_mask = 1 - current_df.isnull().values
                    c_data = ((current_df.fillna(0) - df_full.mean()) / df_full.std()).values
                    n_samples = c_data.shape[0] // eval_length

                    for i in range(n_samples):
                        start_index = i * eval_length
                        end_index = start_index + eval_length

                        c_data_i = c_data[start_index:end_index, :]
                        c_mask_i = c_mask[start_index:end_index, :]
                        real_i = real[start_index:end_index, :]

                        self.observed_data.append(c_data_i)
                        self.observed_masks.append(c_mask_i)
                        self.gt_data.append(real_i)
                        self.gt_mask.append(c_mask_i)

        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_data))
        else:
            self.use_index_list = use_index_list

        # print(self.observed_data[0].shape)

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_data[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_mask[index],
            "timepoints": np.arange(self.eval_length),
            "gt_data": self.gt_data[index],
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(batch_size, device, missing_ratio=0):
    df_full = pd.read_csv("./data/missing{}/{}_missing.csv".format(missing_ratio, missing_ratio))
    df_full = df_full[['x_coordinate', 'y_coordinate', 'z_coordinate']]

    dataset = Human_Dataset(eval_length=36, target_dim=3, use_index_list=None, missing_ratio=missing_ratio)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=1, shuffle=True
    )
    dataset_test = Human_Dataset(eval_length=36, target_dim=3, use_index_list=None, missing_ratio=missing_ratio)
    test_loader = DataLoader(
        dataset_test, batch_size=batch_size, num_workers=1, shuffle=False
    )
    dataset_valid = Human_Dataset(eval_length=36, target_dim=3, use_index_list=None, missing_ratio=missing_ratio)
    valid_loader = DataLoader(
        dataset_valid, batch_size=batch_size, num_workers=1, shuffle=False
    )

    scaler = torch.from_numpy(df_full.std().values).to(device).float()
    mean_scaler = torch.from_numpy(df_full.mean().values).to(device).float()

    return train_loader, valid_loader, test_loader, scaler, mean_scaler
