import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join as join_path
from os.path import exists as is_path_exist

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.optim import lr_scheduler

import sys
sys.path.append('../')
from utils import util
from model.model import MyModel
from dataset.dataset import MyDataset

def train_fn(model: MyModel, dataloader: DataLoader, optimizer: optim, criterion: nn):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader):
        optimizer.zero_grad()

        output = model(*batch)
        loss = criterion(*output)

        loss.backward()
        optimizer.step()

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

def train_loop(cfg, model: MyModel,
                    trainloader: DataLoader,
                    validloader: DataLoader,
                    criterion: nn,
                    optimizer: optim,
                    scheduler: lr_scheduler=None):
    best_valid_loss = np.Inf
    for i in range(cfg['epochs']):
        train_loss = train_fn(model, trainloader, optimizer, criterion)
        valid_loss = valid_fn(model, validloader, criterion)
        if scheduler is not None:
            scheduler.step()
        if valid_loss < best_valid_loss:
            torch.save(model.state_dict(), join_path(cfg['save_model_dir'], f'best_{model.modeltype}.pt'))
            best_valid_loss = valid_loss
            print('SAVED_WEIGHTS success')
        
        print(f'Epoch: {i} \t trainloss: {train_loss} \t validloss: {valid_loss}')

if __name__ == '__main__':
    curr_dir = os.path.dirname(__file__)
    cfg = util.load_cfg(join_path(curr_dir, 'config/configuration.json'))
    cfg['data_dir'] = join_path(curr_dir, cfg['data_dir'])
    cfg['save_model_dir'] = join_path(curr_dir, cfg['save_model_dir'])
    
    df = pd.read_csv(join_path(cfg['data_dir'], "metadata.csv"))
    
    train_df = df[df['sample_id'] < cfg['split-point']]
    valid_df = df[df['sample_id'] >= cfg['split-point']]
    
    trainset = MyDataset(train_df, cfg)
    validset = MyDataset(valid_df, cfg)

    trainloader = DataLoader(trainset, cfg['batch_size'], shuffle=True)
    validloader = DataLoader(validset, cfg['batch_size'])
    print('dataloader created')
    
    model = MyModel(modeltype=cfg['model_type'], emb_size=cfg['emb_size'])
    model_weight_path = join_path(cfg['save_model_dir'], f'best_{model.modeltype}_056.pt')
    if is_path_exist(model_weight_path):
        model.load_state_dict(torch.load(model_weight_path))
        print('loaded saved model')
    model.to(cfg['device'])
    print('model created, type=', model.modeltype)

    criterion = nn.TripletMarginLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['LR'])
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=cfg['scheduler_milestones'], gamma=cfg['scheduler_gamma'])

    train_loop(cfg, model, trainloader, validloader, criterion, optimizer, scheduler)