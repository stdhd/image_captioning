from typing import Callable

import torch.nn as nn
from torchsummary import summary
from torchvision import models


class Encoder(nn.Module):
    def __init__(self, base_arch: Callable, pretrained=True):
        super(Encoder, self).__init__()
        loaded_model = base_arch(pretrained)
        self.features = loaded_model.features[:-1]  # drop MaxPool2d-layer
        self.avgpool = nn.AdaptiveAvgPool2d((14, 14))  # allow input images of variable size (14×14×512 as in paper 4.3)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)  # 512×14×14
        # x = x.view(x.shape[0], 512, -1)  # 512×196
        return x


if __name__ == '__main__':
    model = Encoder(models.vgg16, pretrained=True)
    summary(model, input_size=(3, 224, 224), device='cpu')
