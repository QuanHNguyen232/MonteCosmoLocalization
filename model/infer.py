import os
import numpy as np
import PIL

import torch
import torch.nn as nn

import sys
sys.path.append('../')
from utils import util
from model.model import MyModel

def get_infer_img(img_path: str):
  img = np.asarray(PIL.Image.open(img_path))
  img = np.stack([img, img, img], axis=-1)
  img = torch.from_numpy(img).permute(2, 0, 1).to(cfg['device']) / 255.0
  return img.unsqueeze(0)

def get_emb(model: MyModel, img) -> np.ndarray:
  with torch.no_grad():
    emb = model.forward_one(img)
    return emb.squeeze(0).detach().cpu().numpy()

def euclidean_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  dist = np.linalg.norm(a-b, axis=1)    # along vector of 512
  return dist

def infer(model: MyModel, cfg: dict, sample_id: int, img_id: int) -> np.ndarray:
    img_dir = cfg['data_dir']
    sample_size = len(os.listdir(f'{img_dir}/{sample_id}'))
    with torch.no_grad():
        img_A = get_infer_img(f'{img_dir}/{sample_id}/{img_id}.jpg')
        emb_A = get_emb(model, img_A)

        imgs = [get_infer_img(f'{img_dir}/{sample_id}/{i}.jpg') for i in range(sample_size) if i!=img_id]
        embs = np.array([get_emb(model, img) for img in imgs])

        distances = euclidean_dist(np.expand_dims(emb_A, 0), embs) # (1, 512) vs (71, 512)
        closest_idx = np.argsort(distances)
        return closest_idx

if __name__ == '__main__':
    cfg = util.load_cfg()

    model = MyModel(modeltype='resnet18', emb_size=cfg['emb_size'])
    model_weight_path = f'best_{model.modeltype}.pt'
    if os.path.exists(model_weight_path):
        model.load_state_dict(torch.load(model_weight_path))
        print('loaded saved model')
    model.to(cfg['device'])
    print('model created, type=', model.modeltype)

    result = infer(model, cfg, 17, 8)
    print(result)