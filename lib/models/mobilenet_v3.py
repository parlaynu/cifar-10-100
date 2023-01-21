from itertools import islice

import torch
import torch.nn as nn
import torchvision.models


def _tweak_mobilenet(model, num_classes, use_gpu, freeze):
    
    # update the final classifier
    fc = model.classifier[-1]
    if num_classes is not None and num_classes != fc.out_features:
        newfc = nn.Linear(fc.in_features, num_classes)
        model.classifier.pop(-1)
        model.classifier.append(newfc)
    
    # freeze any layers
    for child in islice(model.features.children(), freeze):
        for idx, p in enumerate(child.parameters()):
            p.requires_grad = False

    # move the model to the device
    device = torch.device('cuda') if (use_gpu and torch.cuda.is_available()) else torch.device('cpu')
    model = model.to(device)

    # set extra attributes on the model
    model.device = device
    model.num_outputs = num_classes
    
    return model


def mobilenet_v3_small(num_classes, *, pretrained=True, use_gpu=False, freeze=0):
    weights = None
    if pretrained:
        weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    
    model = torchvision.models.mobilenet_v3_small(weights=weights)
    model = _tweak_mobilenet(model, num_classes, use_gpu, freeze)

    model.fullname = "torchvision.models.mobilenet_v3_small"

    return model


def mobilenet_v3_large(num_classes, *, pretrained=True, use_gpu=False, freeze=0):
    weights = None
    if pretrained:
        weights = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1

    model = torchvision.models.mobilenet_v3_large(weights=weights)
    model = _tweak_mobilenet(model, num_classes, use_gpu, freeze)

    model.fullname = "torchvision.models.mobilenet_v3_large"

    return model


