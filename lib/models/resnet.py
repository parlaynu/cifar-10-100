from itertools import islice

import torch
import torch.nn as nn
import torchvision.models


def _tweak_resnet(model, num_classes, force_cpu, freeze):

    # update the final classifier
    fc = model.fc
    if num_classes is not None and num_classes != fc.out_features:
        newfc = nn.Linear(fc.in_features, num_classes)
        model.fc = newfc
    
    # freeze layers
    for child in islice(model.children(), freeze):
        for p in child.parameters():
            p.requires_grad = False

    # move the model to the device
    device = torch.device('cuda') if (torch.cuda.is_available() and not force_cpu) else torch.device('cpu')
    model = model.to(device)

    # set extra attributes on the model
    model.device = device
    model.num_outputs = num_classes
    
    return model


def resnet18(num_classes, *, pretrained=True, force_cpu=False, freeze=0):
    weights = None
    if pretrained:
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1

    model = torchvision.models.resnet18(weights=weights)
    model = _tweak_resnet(model, num_classes, force_cpu, freeze)

    model.fullname = "torchvision.models.resnet18"

    return model


def resnet34(num_classes, *, pretrained=True, force_cpu=False, freeze=0):
    weights = None
    if pretrained:
        weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1

    model = torchvision.models.resnet34(weights=weights)
    model = _tweak_resnet(model, num_classes, force_cpu, freeze)

    model.fullname = "torchvision.models.resnet34"

    return model


def resnet50(num_classes, *, pretrained=True, force_cpu=False, freeze=0):
    weights = None
    if pretrained:
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1

    model = torchvision.models.resnet50(weights=weights)
    model = _tweak_resnet(model, num_classes, force_cpu, freeze)

    model.fullname = "torchvision.models.resnet50"

    return model


def resnet101(num_classes, *, pretrained=True, force_cpu=False, freeze=0):
    weights = None
    if pretrained:
        weights = torchvision.models.ResNet101_Weights.IMAGENET1K_V1

    model = torchvision.models.resnet101(weights=weights)
    model = _tweak_resnet(model, num_classes, force_cpu, freeze)

    model.fullname = "torchvision.models.resnet101"

    return model


def resnet152(num_classes, *, pretrained=True, force_cpu=False, freeze=0):
    weights = None
    if pretrained:
        weights = torchvision.models.ResNet152_Weights.IMAGENET1K_V1

    model = torchvision.models.resnet152(weights=weights)
    model = _tweak_resnet(model, num_classes, force_cpu, freeze)

    model.fullname = "torchvision.models.resnet152"

    return model
