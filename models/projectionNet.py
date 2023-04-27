import torch.nn as nn
from torchvision import models

class _ProjectionNet(nn.Module):
    def __init__(self, encoder, head_dims, classes = 2, pretrained = True):
        super().__init__()
        self.encoder = getattr( models, encoder)(pretrained = pretrained)
        last_layer = list(self.encoder.named_modules())[-1][0].split('.')[0]
        setattr(self.encoder, last_layer, nn.Identity())
        head = []
        for d in head_dims[:-1]:
            head.append(nn.Linear(d, d, bias=False)),
            head.append(nn.BatchNorm1d(d))
            head.append(nn.ReLU(inplace=True))
        embeds = nn.Linear(head_dims[-2], head_dims[-1], bias=classes)
        head.append(embeds)
        self.head = nn.Sequential(*head)
        self.out = nn.Linear(head_dims[-1], classes)

    def forward(self, x):
        features = self.encoder(x)
        embeds = self.head(features)
        logits = self.out(embeds)
        return (logits, embeds)

    def freeze_resnet(self, layer_name):
        check = True
        for name, param in self.encoder.named_parameters():
            if name == layer_name:
                check = True
            if not check and param.requires_grad != False:
                param.requires_grad = False
            else:
                param.requires_grad = True