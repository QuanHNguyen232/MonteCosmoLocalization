import torch
import torch.nn as nn

from torchvision.models import resnet50, ResNet50_Weights

class MyModel(nn.Module):
    def __init__(self, emb_size=512):
        super(MyModel, self).__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        # self.efficientnet.classifier = nn.Linear(in_features=self.efficientnet.classifier.in_features, out_features=emb_size)

    def forward(self, x):
        x = self.backbone(x)
        return x