"""
code-1
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import pickle
import time
import os
import argparse

from tars.tars_data_loaders import *
from tars.tars_training import *
from tars.tars_model import *

parser = argparse.ArgumentParser(description='Training a pytorch model to classify different plants')
parser.add_argument('-idl', '--input_data_loc', help='', default='data/training_data')
parser.add_argument('-mo', '--model_name', default="resnet18")
parser.add_argument('-f', '--freeze_layers', default=False)
parser.add_argument('-ep', '--epochs', default=100, type=int)
parser.add_argument('-b', '--batch_size', default=32, type=int)
parser.add_argument('-sl', '--save_loc', default="models/" )
parser.add_argument("-g", '--use_gpu',  default=False)

args = parser.parse_args()


dataloaders, dataset_sizes, class_names = generate_data(args.input_data_loc, batch_size=args.batch_size)
print(class_names)

print("[Load the model...]")
# Parameters of newly constructed modules have requires_grad=True by default
model_conv = all_pretrained_models(len(class_names), use_gpu=args.use_gpu, freeze_layers=args.freeze_layers, name=args.model_name)

print("[Using CrossEntropyLoss...]")
criterion = nn.CrossEntropyLoss()


print("[Using small learning rate with momentum...]")
# Observe that all parameters are being optimized
#optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)


print("[Creating Learning rate scheduler...]")
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

print("[Training the model begun ....]")
model_ft = train_model(model_conv, dataloaders, dataset_sizes, criterion, optimizer_conv, exp_lr_scheduler, args.use_gpu,
                       num_epochs=args.epochs)

if args.freeze_layers:
    args.model_name = args.model_name+"_freeze"

torch.save(model_ft.state_dict(), args.save_loc+args.model_name+".pth")
