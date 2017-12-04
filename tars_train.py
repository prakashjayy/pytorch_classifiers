"""
code-1
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import os
import argparse

from tars.tars_data_loaders import *
from tars.tars_training import *
from tars.tars_model import *

parser = argparse.ArgumentParser(description='Training a pytorch model to classify different plants')
parser.add_argument('-idl', '--input_data_loc', help='', default='data/training_data')
parser.add_argument('-mo', '--model_name', default="resnet18")
parser.add_argument('-f', '--freeze_layers', default=True, action='store_false', help='Bool type')
parser.add_argument('-ep', '--epochs', default=100, type=int)
parser.add_argument('-b', '--batch_size', default=16, type=int)
parser.add_argument('-is', '--input_shape', default=224, type=int)
parser.add_argument('-sl', '--save_loc', default="models/" )
parser.add_argument("-g", '--use_gpu', default=True, action='store_false', help='Bool type gpu')

args = parser.parse_args()


dataloaders, dataset_sizes, class_names = generate_data(args.input_data_loc, args.input_shape, batch_size=args.batch_size)
print(class_names)

print("[Load the model...]")
# Parameters of newly constructed modules have requires_grad=True by default
print("Loading model using class: {}, use_gpu: {}, freeze_layers: {}, name_of_model: {}".format(len(class_names), args.use_gpu, args.freeze_layers, args.model_name))
model_conv = all_pretrained_models(len(class_names), use_gpu=args.use_gpu, freeze_layers=args.freeze_layers, name=args.model_name)
model_conv = nn.DataParallel(model_conv, device_ids=[0, 1])

print("[Using CrossEntropyLoss...]")
criterion = nn.CrossEntropyLoss()


print("[Using small learning rate with momentum...]")
# Observe that all parameters are being optimized
#optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)

if args.freeze_layers:
    print("[Model freezed. Only training the last layer]")
    if args.model_name in ["densenet121", "densenet161", "densenet169", "densenet201"]:
        print("[Optimizer for Densenet]")
        optimizer_conv = optim.SGD(model_conv.classifier.parameters(), lr=0.001, momentum=0.9)
    elif args.model_name in ["vgg11", "vgg13", "vgg16", "vgg19", "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn", "alexnet"]:
        print("[Optimizer for VGG or Alexnet]")
        optimizer_conv = optim.SGD(list(model_conv.classifier[-1].parameters()), lr=0.001, momentum=0.9)
    elif args.model_name in ["squeezenet1_0", "squeezenet1_1"]:
        print("[Optimizer for squeezenet]")
        optimizer_conv = optim.SGD(list(model_conv.classifier.parameters()), lr=0.001, momentum=0.9)
    else:
        print("[Optimizer for inception or resnet]")
        optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

else:
    print("[Fine tuning the entire module]")
    optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)


print("[Creating Learning rate scheduler...]")
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

print("[Training the model begun ....]")
model_ft = train_model(model_conv, dataloaders, dataset_sizes, criterion, optimizer_conv, exp_lr_scheduler, args.use_gpu,
                       num_epochs=args.epochs)

model_save_loc = args.save_loc+args.model_name+"_"+str(args.freeze_layers)+"_freeze"+".pth"

torch.save(model_ft.state_dict(), model_save_loc)
