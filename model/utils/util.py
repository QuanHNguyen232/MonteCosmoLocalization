import os
import cv2
import json
# import torch

# def load_cfg(filename: str='./config/configuration.json'):
#     with open(filename, 'r') as jsonfile:
#         cfg = json.load(jsonfile)
#         cfg['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
#         print('load_cfg SUCCESS')
#         return cfg

def get_img(img_path: str):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img