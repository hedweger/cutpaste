import torch.nn as nn
from models.projectionNet import _ProjectionNet

class resnet18(_ProjectionNet):
    def  __init__(self, encoder = 'resnet18', pretrained = False):
        super().__init__(encoder = encoder, head_dims=[512,128], pretrained = pretrained)
        return
    
class resnet50(_ProjectionNet):
    def  __init__(self, encoder = 'resnet50', pretrained = False):
        super().__init__(encoder = encoder, head_dims = [2048,128], pretrained = pretrained)
        return
    
class vgg(_ProjectionNet):
    def  __init__(self, encoder = 'vgg11', pretrained = False):
        super().__init__(encoder = encoder, head_dims=[25088,128], pretrained = pretrained)
        return

class densenet(_ProjectionNet):
    def  __init__(self, encoder = 'densenet121', pretrained = False):
        super().__init__(encoder = encoder, head_dims=[1024,128], pretrained = pretrained)
        return
    
class efficientnet(_ProjectionNet):
    def  __init__(self, encoder = 'efficientnet_b1', pretrained = False):
        super().__init__(encoder = encoder, head_dims=[1280,128], pretrained = pretrained)
        return