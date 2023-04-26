import torch
import torch.nn as nn

from torchvision.models import vgg16_bn, VGG16_BN_Weights

import sys
sys.path.append('../')
from utils.util import load_cfg

class MyModel(nn.Module):
    def __init__(self, emb_size=512):
        super(MyModel, self).__init__()
        self.backbone = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
        self.backbone.classifier[3] = nn.Linear(in_features=self.backbone.classifier[3].in_features,
                                                out_features=self.backbone.classifier[3].in_features // 2)
        self.backbone.classifier[6] = nn.Linear(in_features=self.backbone.classifier[3].out_features,
                                                out_features=emb_size)
        self.freeze_layer()

    def forward_one(self, x):
        x = self.backbone(x)
        return x
    
    def forward(self, a, p, n):
        a = self.forward_one(a)
        p = self.forward_one(p)
        n = self.forward_one(n)
        return a, p, n
    
    def freeze_layer(self):
        for name, param in self.backbone.features[:34].named_parameters():
            param.requires_grad = False

if __name__ == '__main__':
    cfg = load_cfg('../config/configuration.json')
    model = MyModel(cfg['emb_size'])
    print(model)
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)