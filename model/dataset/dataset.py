import os
import random
import math
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from utils.util import get_img

DATA_DIR = 'D:\Coding-Workspace\MonteCosmoLocalization\model\data'
SAMPLE_SIZE = 72    # 72imgs/360 degree

def MyDataset(Dataset):
    def __init__(self, df_path):
        self.df = pd.read_csv(df_path)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        A_id = row.image_id
        P_id = (A_id + (1 if random.random() < 0.5 else -1)) % SAMPLE_SIZE
        N_id = random.choice(list(set(range(SAMPLE_SIZE)) - set([A_id-1, A_id, A_id+1])))
        
        A_img = get_img(os.path.join(DATA_DIR, f'{row.sample_id}/{A_id}.jpg'))
        P_img = get_img(os.path.join(DATA_DIR, f'{row.sample_id}/{P_id}.jpg'))
        N_img = get_img(os.path.join(DATA_DIR, f'{row.sample_id}/{N_id}.jpg'))

        A_img = torch.from_numpy(A_img).permute(2, 0, 1) / 255.0  # permute: (h, w, c)->(c, h, w)
        P_img = torch.from_numpy(P_img).permute(2, 0, 1) / 255.0
        N_img = torch.from_numpy(N_img).permute(2, 0, 1) / 255.0
        
        return A_img, P_img, N_img
