""" data loaders

Given a input folder of train and valid, this should periodically generate batches of input images with labels

Important Augmentations: https://github.com/pytorch/vision/blob/master/torchvision/transforms/transforms.py
 """

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
from tars.utils import *

def generate_data(data_dir, input_shape,  name, batch_size=32):
    """
    input_shape(scale, shape)
    """
    if name in ["inceptionv4", "inceptionresnetv2", "inception_v3"]:
        scale = 360
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    elif name == "bninception":
        scale = 256
        mean = [104, 117, 128]
        std =  [1, 1, 1]

    elif name == "vggm":
        scale = 256
        mean = [123.68, 116.779, 103.939]
        std = [1, 1, 1]

    elif name == "nasnetalarge":
        scale = 354
        mean = [0.5, 0.5, 0.5]
        std = [1, 1, 1]

    else:
        scale = 256
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    print("[Scale: {} , mean: {}, std: {}]".format(scale, mean, std))
    if name == "bninception":
        print("[Using bninception Dataloader]")
        data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(scale),
            transforms.RandomResizedCrop(input_shape),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=90),
            transforms.ToTensor(),
            ToSpaceBGR(True),
            ToRange255(True),
            transforms.Normalize(mean, std)]),
        'val': transforms.Compose([
            transforms.Resize(scale),
            transforms.CenterCrop(input_shape),
            transforms.ToTensor(),
            ToSpaceBGR(True),
            ToRange255(True),
            transforms.Normalize(mean, std)]),}
    else:
        data_transforms = {
        'train': transforms.Compose([
        transforms.Resize(scale),
        transforms.RandomResizedCrop(input_shape),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=90),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),
        'val': transforms.Compose([
        transforms.Resize(scale),
        transforms.CenterCrop(input_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),}
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=4) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    return dataloaders, dataset_sizes, class_names




def generate_data_simple_agumentation(data_dir, input_shape,  name, batch_size=32):
    """
    input_shape(scale, shape)
    """
    if name in ["inceptionv4", "inceptionresnetv2", "inception_v3"]:
        scale = 360
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    elif name == "bninception":
        scale = 256
        mean = [104, 117, 128]
        std =  [1, 1, 1]

    elif name == "vggm":
        scale = 256
        mean = [123.68, 116.779, 103.939]
        std = [1, 1, 1]

    elif name == "nasnetalarge":
        scale = 354
        mean = [0.5, 0.5, 0.5]
        std = [1, 1, 1]

    else:
        scale = 256
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    print("[Scale: {} , mean: {}, std: {}]".format(scale, mean, std))
    if name == "bninception":
        print("[Using bninception Dataloader]")
        data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(scale),
            transforms.RandomResizedCrop(input_shape),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ToSpaceBGR(True),
            ToRange255(True),
            transforms.Normalize(mean, std)]),
        'val': transforms.Compose([
            transforms.Resize(scale),
            transforms.CenterCrop(input_shape),
            transforms.ToTensor(),
            ToSpaceBGR(True),
            ToRange255(True),
            transforms.Normalize(mean, std)]),}
    else:
        data_transforms = {
        'train': transforms.Compose([
        transforms.Resize(scale),
        transforms.RandomResizedCrop(input_shape),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),
        'val': transforms.Compose([
        transforms.Resize(scale),
        transforms.CenterCrop(input_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),}
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=4) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    return dataloaders, dataset_sizes, class_names


def data_loader_predict(data_dir, input_shape, name):
    if name in ["inceptionv4", "inceptionresnetv2", "inception_v3"]:
        scale = 360
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    elif name == "bninception":
        scale = 256
        mean = [104, 117, 128]
        std =  [1, 1, 1]

    elif name == "vggm":
        scale = 256
        mean = [123.68, 116.779, 103.939]
        std = [1, 1, 1]

    elif name == "nasnetalarge":
        scale = 354
        mean = [0.5, 0.5, 0.5]
        std = [1, 1, 1]

    else:
        scale = 256
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    print("[Scale: {} , mean: {}, std: {}]".format(scale, mean, std))
    if name == "bninception":
        val = transforms.Compose([transforms.Scale(scale),
                              transforms.TenCrop(input_shape),
                              transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                              transforms.Lambda(lambda bgr: torch.stack([ToSpaceBGR(True)(bgrformat) for bgrformat in bgr])),
                              transforms.Lambda(lambda range255: torch.stack([ToRange255(True)(ranges) for ranges in range255])),
                              transforms.Lambda(lambda normal: torch.stack([transforms.Normalize(mean, std)(normalize) for normalize in normal]))])
    else:
        val = transforms.Compose([transforms.Scale(scale),
                              transforms.TenCrop(input_shape),
                              transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                              transforms.Lambda(lambda normal: torch.stack([transforms.Normalize(mean, std)(normalize) for normalize in normal]))])
    image_datasets = datasets.ImageFolder(data_dir, val)
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=1,
                                         shuffle=False, num_workers=1)
    return dataloaders, image_datasets
