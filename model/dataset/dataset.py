import os
import random
import cv2
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import PIL

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

import sys
sys.path.append('../')

class MyDataset(Dataset):
    def __init__(self, df, cfg):
        self.cfg = cfg
        self.data_dir = cfg['data_dir']
        self.device = cfg['device']
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        A_id = row['image_id']
        P_id = (A_id + random.choice([-2, -1, 1, 2])) % row['sample_size']
        N_id = random.choice(list(set(range(row['sample_size'])) - set([A_id-2, A_id-1, A_id, A_id+1, A_id+2])))
        
        A_img = self.get_img(os.path.join(self.data_dir, f'{row.sample_id}/{A_id}.jpg'))
        P_img = self.get_img(os.path.join(self.data_dir, f'{row.sample_id}/{P_id}.jpg'))
        N_img = self.get_img(os.path.join(self.data_dir, f'{row.sample_id}/{N_id}.jpg'))
        
        return A_img, P_img, N_img
    
    @staticmethod
    def get_img(self, img_path: str):
      img = np.asarray(PIL.Image.open(img_path))    # using PIL get imgs faster than opencv
      img = np.stack([img, img, img], axis=-1)
      img = torch.from_numpy(img).permute(2, 0, 1).to(self.device) / 255.0
      return img

if __name__ == '__main__':
    from utils.util import load_cfg

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