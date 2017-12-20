""" Given the pre-trained model and model_weights this will predict the output for the code

python tars_predict.py -idl data/training_data -ll models/resnet18_True_freeze.pth -mo "resnet18" -nc 12 -is 224
"""

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

import pandas as pd
from collections import OrderedDict

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
parser.add_argument("-gi", '--used_dataparallel', default=True, action='store_false', help='Bool type gpu')
parser.add_argument("-nc", '--num_classes', default=12, type=int)

args = parser.parse_args()

if not os.path.exists(args.save_loc):
    os.makedirs(args.save_loc)

print("use_gpu: {}, name_of_model: {}".format(args.use_gpu, args.model_name))
model_conv = all_pretrained_models(args.num_classes, use_gpu=args.use_gpu, name=args.model_name)

print("[Loading the pretrained model on this datasets]")
state_dict = torch.load(args.load_loc)

if args.used_dataparallel:
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    print("[Loading Weights to the Model]")
    model_conv.load_state_dict(new_state_dict)

if not args.used_dataparallel:
    model_conv = nn.DataParallel(model_conv, device_ids=[0, 1, 2])
    model_conv.load_state_dict(state_dict)

model_conv = model_conv.eval()




print("[Validating on Train data]")
train_predictions, class_to_idx = model_evaluation("train", model_conv, args.input_data_loc, args.input_shape, args.use_gpu, args.model_name)
idx_to_class = {class_to_idx[i]:i for i in class_to_idx.keys()}
train_predictions["predicted_class_name"] = train_predictions["predicted"].apply(lambda x: idx_to_class[x])
save_loc = args.save_loc+args.load_loc.rsplit("/")[-1].rsplit(".")[0]+"_train_"+".csv"
train_predictions.to_csv(save_loc, index=False)

print("[Validating on Validating data]")
val_predictions, class_to_idx = model_evaluation("val", model_conv, args.input_data_loc, args.input_shape, args.use_gpu, args.model_name)
idx_to_class = {class_to_idx[i]:i for i in class_to_idx.keys()}
val_predictions["predicted_class_name"] = val_predictions["predicted"].apply(lambda x: idx_to_class[x])
save_loc = args.save_loc+args.load_loc.rsplit("/")[-1].rsplit(".")[0]+"_val_"+".csv"
val_predictions.to_csv(save_loc, index=False)

print("[Predictions on Test data]")
predictions = model_evaluation_test("test", model_conv, "data/Test", args.input_shape, args.use_gpu, args.model_name)
predictions["predicted_class_name"] = predictions["predicted"].apply(lambda x: idx_to_class[x])
save_loc = args.save_loc+args.load_loc.rsplit("/")[-1].rsplit(".")[0]+"_test_"+".csv"
predictions.to_csv(save_loc, index=False)
