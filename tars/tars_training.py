""" training functions
"""

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
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import pandas as pd

from tars.tars_data_loaders import *

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, use_gpu, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in tqdm(dataloaders[phase]):
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                if type(outputs) == tuple:
                    outputs, _ = outputs
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def predicted_class(mode, model_conv, input_data_loc, input_shape, use_gpu):
    print("[Evaluating the {} data]".format(mode))
    original, predicted, probs = [], [], []
    if mode in ["train", "val"]:
        data_loc = os.path.join(input_data_loc, mode)
    else:
        data_loc = "data/Test"
    dataloaders, image_datasets = data_loader_predict(data_loc, input_shape)
    if mode in ["train", "val"]:
        class_to_idx = image_datasets.class_to_idx
    imgs =[i[0] for i in image_datasets.imgs]
    print("total number of {} images: {}".format(mode, len(imgs)))
    for img, label in dataloaders:
        if use_gpu:
            inputs = Variable(img.cuda())
        else:
            inputs = Variable(img)
        outputs = model_conv(inputs)
        if type(outputs) == tuple:
            outputs, _ = outputs
        else:
            outputs=outputs
        outputs = nn.Softmax()(outputs)
        prob, preds = torch.max(outputs.data, 1)
        probs.extend(prob.cpu().numpy())
        original.extend(label.numpy())
        predicted.extend(preds.cpu().numpy())
    if mode in ["train", "val"]:
        print("Accuracy_score {} : {} ".format(mode,  accuracy_score(original, predicted)))
        return class_to_idx
    else:
        print(len(predicted))
        frame = pd.DataFrame(predicted)
        frame.columns = ["class"]
        frame["Prob"] = probs
        frame["img_loc"] = imgs
        frame = frame[["img_loc", "Prob", "class"]]
        return frame
