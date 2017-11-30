""" Models
"""
import torch
import torchvision
import torch.nn as nn
import numpy as np

"""
resnet18, resnet34, resnet50, resnet101, resnet152
densenet121, densenet161, densenet169, densenet201
squeezenet1_0, squeezenet1_1
alexnet,
inception_v3,
vgg11, vgg13, vgg16, vgg19
vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn

"""

def all_pretrained_models(n_class, use_gpu=True, freeze_layers=False, name="resnet18"):
    if name == "alexnet":
        model_conv = torchvision.models.alexnet(pretrained='imagenet')
    if name == "inception_v3":
        model_conv = torchvision.models.inception_v3(pretrained='imagenet')
    if name == "resnet18":
        model_conv = torchvision.models.resnet18(pretrained='imagenet')
    if name == "resnet34":
        model_conv = torchvision.models.resnet34(pretrained='imagenet')
    if name == "resnet50":
        model_conv = torchvision.models.resnet50(pretrained='imagenet')
    if name == "resnet101":
        model_conv = torchvision.models.resnet101(pretrained='imagenet')
    if name == "resnet152":
        model_conv = torchvision.models.resnet152(pretrained='imagenet')
    if name == "densenet121":
        model_conv = torchvision.models.densenet121(pretrained='imagenet')
    if name == "densenet169":
        model_conv = torchvision.models.densenet169(pretrained='imagenet')
    if name == "densenet201":
        model_conv = torchvision.models.densenet201(pretrained='imagenet')
    if name == "squeezenet1_0":
        model_conv = torchvision.models.squeezenet1_0(pretrained='imagenet')
    if name == "squeezenet1_1":
        model_conv = torchvision.models.squeezenet1_1(pretrained='imagenet')
    if name == "vgg11":
        model_conv = torchvision.models.vgg11(pretrained='imagenet')
    if name == "vgg13":
        model_conv = torchvision.models.vgg13(pretrained='imagenet')
    if name == "vgg16":
        model_conv = torchvision.models.vgg16(pretrained='imagenet')
    if name == "vgg19":
        model_conv = torchvision.models.vgg19(pretrained='imagenet')
    if name == "vgg13_bn":
        model_conv = torchvision.models.vgg13_bn(pretrained='imagenet')
    if name == "vgg16_bn":
        model_conv = torchvision.models.vgg16_bn(pretrained='imagenet')
    if name == "vgg19_bn":
        model_conv = torchvision.models.vgg19_bn(pretrained='imagenet')

    if freeze_layers:
        for param in model_conv.parameters():
            param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, n_class)
    if use_gpu:
        model_conv = model_conv.cuda()
    return model_conv
