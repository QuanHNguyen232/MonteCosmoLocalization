import os
import random
import math
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

import sys
sys.path.append('../')
from utils.util import load_cfg
from utils.util import get_img

class MyDataset(Dataset):
    def __init__(self, df, cfg):
        self.cfg = cfg
        self.data_dir = cfg['data_dir']
        self.device = cfg['device']
        self.img_size = cfg['img_size']
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_size = (self.img_size, self.img_size)

        A_id = row['image_id']
        P_id = (A_id + (1 if random.random() < 0.5 else -1)) % row['sample_size']
        N_id = random.choice(list(set(range(row['sample_size'])) - set([A_id-1, A_id, A_id+1])))
        
        A_img = get_img(os.path.join(self.data_dir, f'{row.sample_id}/{A_id}.jpg'), img_size)
        P_img = get_img(os.path.join(self.data_dir, f'{row.sample_id}/{P_id}.jpg'), img_size)
        N_img = get_img(os.path.join(self.data_dir, f'{row.sample_id}/{N_id}.jpg'), img_size)
        
        A_img = torch.from_numpy(A_img).permute(2, 0, 1) / 255.0  # permute: (h, w, c)->(c, h, w)
        P_img = torch.from_numpy(P_img).permute(2, 0, 1) / 255.0
        N_img = torch.from_numpy(N_img).permute(2, 0, 1) / 255.0
        return A_img.to(self.device), P_img.to(self.device), N_img.to(self.device)


if __name__ == '__main__':
    cfg = load_cfg('../config/configuration.json')
    # dataset = MyDataset(cfg)

    # a, p, n = dataset[351]

    # plt.imsave('a.jpg', a.detach().cpu().permute(1, 2, 0).numpy())
    # plt.imsave('p.jpg', p.detach().cpu().permute(1, 2, 0).numpy())
    # plt.imsave('n.jpg', n.detach().cpu().permute(1, 2, 0).numpy())
    df = pd.read_csv("/Accounts/turing/students/s24/nguyqu03/Desktop/MonteCosmoLocalization/model/data/metadata.csv")
    
    train_df = df[df['sample_id'] < cfg['split-point']]
    valid_df = df[df['sample_id'] >= cfg['split-point']]

    print(len(train_df))
    print(len(valid_df))