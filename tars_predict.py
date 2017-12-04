""" Given the pre-trained model and model_weights this will predict the output for the code

python tars_predict.py -idl data/training_data -ll models/resnet18_True_freeze.pth -mo "resnet18" -nc 12 -is 224
"""

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

import pandas as pd

import os
import argparse

from tars.tars_data_loaders import *
from tars.tars_training import *
from tars.tars_model import *

parser = argparse.ArgumentParser(description='Inference for a trained PyTorch Model')
parser.add_argument('-idl', '--input_data_loc', help='', default='data/training_data')
parser.add_argument('-is', '--input_shape', default=224, type=int)
parser.add_argument('-ll', '--load_loc', default="models/resnet18_True_freeze.pth" )
parser.add_argument('-mo', '--model_name', default="resnet18")
parser.add_argument('-sl', '--save_loc', default="submission/")
parser.add_argument("-g", '--use_gpu', default=True, action='store_false', help='Bool type gpu')
parser.add_argument("-nc", '--num_classes', default=12, type=int)

args = parser.parse_args()

if not os.path.exists(args.save_loc):
    os.makedirs(args.save_loc)

print("use_gpu: {}, name_of_model: {}".format(args.use_gpu, args.model_name))
model_conv = all_pretrained_models(args.num_classes, use_gpu=args.use_gpu, name=args.model_name)

print("[Loading the pretrained model on this datasets]")
model_conv.load_state_dict(torch.load(args.load_loc))
model_conv = model_conv.eval()

predicted_class("train", model_conv, args.input_data_loc, args.input_shape, args.use_gpu)
class_to_idx = predicted_class("val", model_conv, args.input_data_loc, args.input_shape, args.use_gpu)
frame = predicted_class("test", model_conv, "data/Test", args.input_shape, args.use_gpu)
idx_to_class = {class_to_idx[i]:i for i in class_to_idx.keys()}
frame["label"] = frame["class"].apply(lambda x: idx_to_class[x])
save_loc = args.save_loc+args.load_loc.rsplit("/")[-1].rsplit(".")[0]+".csv"
frame.to_csv(save_loc, index=False)
