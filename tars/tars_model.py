""" Models
"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init
import numpy as np

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
- If you want to train the whole network
"""

def all_pretrained_models(n_class, use_gpu=True, freeze_layers=False, name="resnet18"):
    if name == "alexnet":
        print("[Building alexnet]")
        model_conv = torchvision.models.alexnet(pretrained='imagenet')
    elif name == "inception_v3":
        print("[Building inception_v3]")
        model_conv = torchvision.models.inception_v3(pretrained='imagenet')
    elif name == "resnet18":
        print("[Building resnet18")
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
    elif "squeezenet" in name:
        print("[Building Squeezenet]")
        in_ftrs = model_conv.classifier[1].in_channels
        out_ftrs = model_conv.classifier[1].out_channels
        features = list(model_conv.classifier.children())
        features[1] = nn.Conv2d(in_ftrs, n_class, kernel_size=(2, 2), stride=(1, 1))
        features[3] = nn.AvgPool2d(12, stride=1)
        model_conv.classifier = nn.Sequential(*features)
        model_conv.num_classes = n_class

    elif "vgg" in name or "alexnet" in name:
        print("[Building VGG or Alexnet classifier]")
        num_ftrs = model_conv.classifier[6].in_features
        features = list(model_conv.classifier.children())[:-1]
        features.extend([nn.Linear(num_ftrs, n_class)])
        model_conv.classifier = nn.Sequential(*features)
    else:
        print("[Building inception_v3 or Resnet]")
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = nn.Linear(num_ftrs, n_class)

    if use_gpu:
        model_conv = model_conv.cuda()
    else:
        model_conv = model_conv
    return model_conv
