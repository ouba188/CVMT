# Copyright (c) OpenMMLab. All rights reserved.
from .detectors_resnet import DetectoRS_ResNet
from .resnet import ResNet, ResNetV1d
from .resnet_CVMT import ResNet50_CVMT

__all__ = [
     'ResNet', 'ResNetV1d',
   'DetectoRS_ResNet','ResNet50_CVMT'
]
