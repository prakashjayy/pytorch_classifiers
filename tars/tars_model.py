""" Models
"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init

import numpy as np

from tars.nasnet import nasnetalarge
from tars.resnext import resnext101_32x4d, resnext101_64x4d
from tars.inceptionresnetv2 import inceptionresnetv2
from tars.inceptionv4 import inceptionv4
from tars.bninception import bninception

"""
resnet18, resnet34, resnet50, resnet101, resnet152
densenet121, densenet161, densenet169, densenet201
squeezenet1_0, squeezenet1_1
alexnet,
inception_v3,
vgg11, vgg13, vgg16, vgg19
vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn


Its quite messed up or I messed it a bit.

- If you want to freeze the layers by its name

for name, params in model_conv.named_parameters():
    if name =! "something":
        params.requires_grad=False

- If you want to freeze the first few layers

```
model_ft = models.resnet50(pretrained=True)
ct = 0
for child in model_ft.children():
ct += 1
if ct < 7:
    for param in child.parameters():
        param.requires_grad = False
```

ct = []
for name, child in model_conv.named_children():
    if "layer1" in ct:
        execute this
    ct.append(name)


"""

def all_pretrained_models(n_class, use_gpu=True, freeze_layers=False, freeze_initial_layers=False, name="resnet18"):
    if name == "alexnet":
        print("[Building alexnet]")
        model_conv = torchvision.models.alexnet(pretrained='imagenet')
    elif name == "inception_v3":
        print("[Building inception_v3]")
        model_conv = torchvision.models.inception_v3(pretrained='imagenet')
    elif name == "inceptionresnetv2":
        print("[Building InceptionResnetV2]")
        model_conv = inceptionresnetv2(num_classes=1000, pretrained='imagenet')
    elif name == "inceptionv4":
        print("[Building Inceptionv4]")
        model_conv = inceptionv4(num_classes=1000, pretrained='imagenet')
    elif name == "bninception":
        print("[Building bninception]")
        model_conv = bninception(num_classes=1000, pretrained='imagenet')
    elif name == "resnet18":
        print("[Building resnet18]")
        model_conv = torchvision.models.resnet18(pretrained='imagenet')
    elif name == "resnet34":
        print("[Building resnet34]")
        model_conv = torchvision.models.resnet34(pretrained='imagenet')
    elif name == "resnet50":
        print("[Building resnet50]")
        model_conv = torchvision.models.resnet50(pretrained='imagenet')
    elif name == "resnet101":
        print("[Building resnet101]")
        model_conv = torchvision.models.resnet101(pretrained='imagenet')
    elif name == "resnet152":
        print("[Building resnet152]")
        model_conv = torchvision.models.resnet152(pretrained='imagenet')
    elif name == "densenet121":
        print("[Building densenet121]")
        model_conv = torchvision.models.densenet121(pretrained='imagenet')
    elif name == "densenet169":
        print("[Building densenet169]")
        model_conv = torchvision.models.densenet169(pretrained='imagenet')
    elif name == "densenet201":
        print("[Building densenet201]")
        model_conv = torchvision.models.densenet201(pretrained='imagenet')
    elif name == "squeezenet1_0":
        print("[Building squeezenet1_0]")
        model_conv = torchvision.models.squeezenet1_0(pretrained='imagenet')
    elif name == "squeezenet1_1":
        print("[Building squeezenet1_1]")
        model_conv = torchvision.models.squeezenet1_1(pretrained='imagenet')
    elif name == "vgg11":
        print("[Building vgg11]")
        model_conv = torchvision.models.vgg11(pretrained='imagenet')
    elif name == "vgg13":
        print("[Building vgg13]")
        model_conv = torchvision.models.vgg13(pretrained='imagenet')
    elif name == "vgg16":
        print("[Building vgg16]")
        model_conv = torchvision.models.vgg16(pretrained='imagenet')
    elif name == "vgg19":
        print("[Building vgg19]")
        model_conv = torchvision.models.vgg19(pretrained='imagenet')
    elif name == "vgg11_bn":
        print("[Building vgg11_bn]")
        model_conv = torchvision.models.vgg11_bn(pretrained='imagenet')
    elif name == "vgg13_bn":
        print("[Building vgg13_bn]")
        model_conv = torchvision.models.vgg13_bn(pretrained='imagenet')
    elif name == "vgg16_bn":
        print("[Building vgg16_bn]")
        model_conv = torchvision.models.vgg16_bn(pretrained='imagenet')
    elif name == "vgg19_bn":
        print("[Building vgg19_bn]")
        model_conv = torchvision.models.vgg19_bn(pretrained='imagenet')
    elif name == "nasnetalarge":
        print("[Building nasnetalarge]")
        model_conv = nasnetalarge(num_classes=1000, pretrained="imagenet")
    elif name == "resnext101_64x4d":
        print("[Building resnext101_64x4d]")
        model_conv = resnext101_64x4d(pretrained="imagenet")
    elif name == "resnext101_32x4d":
        print("[Building resnext101_32x4d]")
        model_conv = resnext101_32x4d(pretrained="imagenet")
    else:
        raise ValueError

    if freeze_layers:
        for i, param in model_conv.named_parameters():
            param.requires_grad = False
    else:
        print("[All layers will be trained]")
    # Parameters of newly constructed modules have requires_grad=True by default
    if "densenet" in name:
        print("[Building Densenet]")
        num_ftrs = model_conv.classifier.in_features
        model_conv.classifier = nn.Linear(num_ftrs, n_class)
        if freeze_initial_layers:
            print("[Densenet: Freeezing layers only till denseblock1 including]")
            ct = []
            for name, child in model_conv.features.named_children():
                if "denseblock1" in ct: #freeze all layers from layer1 inclusive
                    for params in child.parameters():
                        params.requires_grad = True
                ct.append(name)
    elif "squeezenet" in name:
        print("[Building Squeezenet]")
        in_ftrs = model_conv.classifier[1].in_channels
        out_ftrs = model_conv.classifier[1].out_channels
        features = list(model_conv.classifier.children())
        features[1] = nn.Conv2d(in_ftrs, n_class, kernel_size=(2, 2), stride=(1, 1))
        features[3] = nn.AvgPool2d(12, stride=1)
        model_conv.classifier = nn.Sequential(*features)
        model_conv.num_classes = n_class
        if freeze_layers:
            ct = []
            print("[Squeezenet: Freeezing layers only till denseblock1 including]")
            for name, child in model_conv.features.named_children():
                if "3" in ct:
                    for params in child.parameters():
                        params.requires_grad = True
                ct.append(name)

    elif "vgg" in name or "alexnet" in name:
        print("[Building VGG or Alexnet classifier]")
        num_ftrs = model_conv.classifier[6].in_features
        features = list(model_conv.classifier.children())[:-1]
        features.extend([nn.Linear(num_ftrs, n_class)])
        model_conv.classifier = nn.Sequential(*features)
        if freeze_layers:
            ct = []
            print("[Alex or VGG: Freeezing layers only till denseblock1 including]")
            for name, child in model_conv.features.named_children():
                if "5" in ct:
                    for params in child.parameters():
                        params.requires_grad = True
                ct.append(name)

    elif "nasnetalarge" in name:
        print("[Building nasnetalarge]")
        num_ftrs = model_conv.last_linear.in_features
        model_conv.last_linear = nn.Linear(num_ftrs, n_class)
        if freeze_layers:
            ct=[]
            print("[nasnetalarge: Freezing layers till cell_5]")
            for name, child in model_conv.named_children():
                if "cell_5" in ct:
                    for params in child.parameters():
                        params.requires_grad=True
                ct.append(name)

    elif "resnext" in name:
        print("[Building resnext]")
        num_ftrs = model_conv.last_linear.in_features
        model_conv.last_linear = nn.Linear(num_ftrs, n_class)
        if freeze_layers:
            ct = []
            for name, child in model_conv.features.named_children():
                if "4" in ct:
                    for param in child.parameters():
                        param.requires_grad = True
                ct.append(name)

    elif "inceptionresnetv2" in name:
        print("[Building inceptionresnetv2]")
        num_ftrs = model_conv.last_linear.in_features
        model_conv.last_linear = nn.Linear(num_ftrs, n_class)
        if freeze_layers:
            ct = []
            print("[inceptionresnetv2: Freezing layers till maxpool_3a]")
            for name, child in model_conv.named_children():
                if "maxpool_3a" in ct:
                    for params in child.parameters():
                        params.requires_grad=True
                ct.append(name)

    elif "inceptionv4" in name:
        print("[Building inceptionv4]")
        num_ftrs = model_conv.last_linear.in_features
        model_conv.last_linear = nn.Linear(num_ftrs, n_class)
        if freeze_layers:
            ct = []
            for name, child in model_conv.features.named_children():
                if "4" in ct:
                    for param in child.parameters():
                        param.requires_grad = True
                ct.append(name)

    elif "bninception" in name:
        print("[Building bninception]")
        num_ftrs = model_conv.last_linear.in_features
        model_conv.last_linear = nn.Linear(num_ftrs, n_class)
        if freeze_layers:
            ct = []
            print("[bninception: Freezing layers till pool2_3x3_s2]")
            for name, child in model_conv.named_children():
                if "pool2_3x3_s2" in ct:
                    for params in child.parameters():
                        params.requires_grad=True
                ct.append(name)
    else:
        print("[Building inception_v3 or Resnet]")
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = nn.Linear(num_ftrs, n_class)

        if freeze_initial_layers:
            if "resnet" in name:
                print("[Resnet: Freezing layers only till layer1 including]")
                ct = []
                for name, child in model_conv.named_children():
                    if "layer1" in ct:
                        for params in child.parameters():
                            params.requires_grad = True
                    ct.append(name)
            else:
                print("[Inception: Freezing layers only till layer1 including]")
                ct = []
                for name, child in model_conv.named_children():
                    if "Conv2d_4a_3x3" in ct:
                        for params in child.parameters():
                            params.requires_grad = True
                    ct.append(name)



    if use_gpu:
        model_conv = model_conv.cuda()
    else:
        model_conv = model_conv
    return model_conv
