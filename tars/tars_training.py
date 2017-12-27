""" training functions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms

import numpy as np
import time
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import pandas as pd

from tars.utils import *
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


def train_model_mixup(model, dataloaders, dataset_sizes, class_names, criterion, optimizer, scheduler, use_gpu=True, num_epochs = 25,  mixup_lambda = 0.2, alpha = 0.5):
    print("Using GPU: {}".format(use_gpu))
    since = time.time()
    best_model_wts = model.state_dict()
    best_at_epoch = 0
    best_acc = 0.0

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        print('-'*10)

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)
            running_loss = 0.0
            running_corrects = 0
            mixup = True
            for  idx, data  in tqdm(enumerate(dataloaders[phase])):
                if phase == 'train' and mixup:
                    (input1, target1) = data
                    input2, target2 = (input1.clone(), target1.clone())

                    perm = np.random.permutation(input1.size()[0])

                    for i in range(input2.size()[0]):
                        input2[i] = input1[perm[i]]
                        target2[i] = target1[perm[i]]

                    target1 = onehot(target1, len(class_names))
                    target2 = onehot(target2, len(class_names))
                    _lambda = torch.Tensor(np.random.beta(alpha, alpha, size=input1.size()[0]))
                    _input_lambda = _lambda.view(-1, 1, 1, 1)
                    _target_lambda = _lambda.view(-1, 1)

                    inputs = _input_lambda * input1 + (1 - _input_lambda) * input2
                    labels = _target_lambda * target1 + (1 - _target_lambda) * target2

                else :
                    inputs, labels = data
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs = Variable(inputs)
                    labels = Variable(labels)
                optimizer.zero_grad()

                outputs = model(inputs)
                if type(outputs) == tuple:
                    outputs, _ = outputs
                if mixup and phase == 'train':
                    loss = naive_cross_entropy_loss(outputs, labels)
                else:
                    loss = criterion(outputs, labels)
                _, preds = torch.max(outputs.data, 1)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0]

                if not mixup or phase == 'val':
                    running_corrects += torch.sum(preds == labels.data)
                else:
                    running_corrects = 0
            epoch_loss = running_loss/dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                best_at_epoch = epoch
        print('Epoch Time: {}'.format(time.time() - epoch_start))
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f} at epoch: {}'.format(best_acc, best_at_epoch))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



def model_evaluation(mode, model_conv, input_data_loc, input_shape, use_gpu, name):
    """A function which evaluates the model and outputs the csv of predictions of validation and train.

    mode:
    model_conv:
    input_data_loc:
    input_shape:
    use_gpu:

    Output:
    1) Prints score of train and validation (Write it to a log file)
    """
    print("[Evaluating the data in {}]".format(mode))
    data_loc = os.path.join(input_data_loc, mode)

    print("[Building dataloaders]")
    dataloaders, image_datasets = data_loader_predict(data_loc, input_shape, name)
    class_to_idx = image_datasets.class_to_idx
    imgs = [i[0] for i in image_datasets.imgs]
    print("total number of {} images: {}".format(mode, len(imgs)))
    original, predicted, probs = [], [], []
    for img, label in dataloaders:
        if use_gpu:
            inputs = Variable(img.cuda())
        else:
            inputs = Variable(img)
        bs, ncrops, c, h, w = inputs.data.size()
        output = model_conv(inputs.view(-1, c, h, w)) # fuse batch size and ncrops
        if type(output) == tuple:
            output, _ = output
        else:
            output = output
        outputs = torch.stack([nn.Softmax(dim=0)(i) for i in output])
        outputs = outputs.mean(0)
        _, preds = torch.max(outputs, 0)
        probs.append(outputs.data.cpu().numpy())
        original.extend(label.numpy())
        predicted.extend(preds.data.cpu().numpy())
    print("Accuracy_score {} : {} ".format(mode,  accuracy_score(original, predicted)))
    frame = pd.DataFrame(probs)
    frame.columns = ["class_{}".format(i) for i in frame.columns]
    frame["img_loc"] = imgs
    frame["original"] = original
    frame["predicted"] = predicted
    return frame, class_to_idx


def model_evaluation_test(mode, model_conv, test_input_data_loc, input_shape, use_gpu, name):
    dataloaders, image_datasets = data_loader_predict(test_input_data_loc, input_shape, name)
    imgs =[i[0] for i in image_datasets.imgs]
    print("total number of {} images: {}".format(mode, len(imgs)))
    predicted, probs = [], []
    for img, label in dataloaders:
        if use_gpu:
            inputs = Variable(img.cuda())
        else:
            inputs = Variable(img)
        bs, ncrops, c, h, w = inputs.data.size()
        output = model_conv(inputs.view(-1, c, h, w)) # fuse batch size and ncrops
        if type(output) == tuple:
            output, _ = output
        else:
            output = output
        outputs = torch.stack([nn.Softmax(dim=0)(i) for i in output])
        outputs = outputs.mean(0)
        _, preds = torch.max(outputs, 0)
        probs.append(outputs.data.cpu().numpy())
        predicted.extend(preds.data.cpu().numpy())
    frame = pd.DataFrame(probs)
    frame.columns = ["class_{}".format(i) for i in frame.columns]
    frame["img_loc"] = imgs
    frame["predicted"] = predicted
    return frame
