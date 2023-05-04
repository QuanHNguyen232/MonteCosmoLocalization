import torch
import torch.nn as nn

from torchvision.models import vgg16_bn, VGG16_BN_Weights, resnet18, ResNet18_Weights

import sys
sys.path.append('../')

class MyModel(nn.Module):
    def __init__(self, modeltype='resnet18', emb_size=512):
        super(MyModel, self).__init__()
        self.modeltype = modeltype
        if modeltype == 'resnet18':
          self.backbone = resnet18(ResNet18_Weights.DEFAULT)
          self.backbone.fc = nn.Linear(in_features=self.backbone.fc.in_features, out_features=emb_size)
        else:
          self.modeltype = 'vgg16_bn'
          self.backbone = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
          self.backbone.classifier[6] = nn.Linear(in_features=self.backbone.classifier[6].out_features,
                                                out_features=emb_size)

    def forward_one(self, x):
        x = self.backbone(x)
        return x
    
    def forward(self, a, p, n):
        a = self.forward_one(a)
        p = self.forward_one(p)
        n = self.forward_one(n)
        return a, p, n
    
if __name__ == '__main__':
    from utils.util import load_cfg
    cfg = load_cfg('../config/configuration.json')
    model = MyModel(emb_size=cfg['emb_size'])
    print(model)
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)