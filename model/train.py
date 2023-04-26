import os
import numpy as np
import pandas as pd
from utils import util
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
sys.path.append('../')
from model.model import MyModel
from dataset.dataset import MyDataset

def train_fn(model: MyModel, dataloader: DataLoader, optimizer: torch.optim, criterion: nn):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader):
        optimizer.zero_grad()

        output = model(*batch)
        loss = criterion(*output)

        loss.backward()
        criterion.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def valid_fn(model: MyModel, dataloader: DataLoader, criterion: nn):
    model.eval()
    total_loss = 0.0
    for batch in tqdm(dataloader):
        output = model(*batch)
        loss = criterion(*output)

        total_loss += loss.item()
    return total_loss / len(dataloader)

def train_loop(model, trainloader, validloader, criterion, optimizer, cfg):
    best_valid_loss = np.Inf
    for i in range(cfg['epochs']):
        train_loss = train_fn(model, trainloader, optimizer, criterion)
        valid_loss = valid_fn(model, validloader, criterion)
        if valid_loss < best_valid_loss:
            torch.save(model.state_dict(), 'best_model.pt')
            best_valid_loss = valid_loss
            print('SAVED_WEIGHTS success')
        
        print(f'Epoch: {i} \t trainloss: {train_loss} \t validloss: {valid_loss}')

if __name__ == '__main__':
    cfg = util.load_cfg()

    df = pd.read_csv(os.path.join(cfg['data_dir'], "metadata.csv"))
    
    train_df = df[df['sample_id'] < cfg['split-point']]
    valid_df = df[df['sample_id'] >= cfg['split-point']]
    
    trainset = MyDataset(train_df, cfg)
    validset = MyDataset(valid_df, cfg)

    trainloader = DataLoader(trainset, cfg['batch_size'], shuffle=True)
    validloader = DataLoader(validset, cfg['batch_size'])
    print('dataloader created')
    
    model = MyModel(cfg['emb_size'])
    model.to(cfg['device'])
    print('model created')

    criterion = nn.TripletMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['LR'])
    
    with torch.no_grad:
        for batch in tqdm(trainloader):
            break
        output = model(*batch)
    # train_loop(model, trainloader, validloader, criterion, optimizer, cfg)