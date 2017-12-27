import torch
import torch.nn as nn
from torch.nn import functional as Func
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np

class ToSpaceBGR(object):

    def __init__(self, is_bgr):
        self.is_bgr = is_bgr

    def __call__(self, tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor


class ToRange255(object):

    def __init__(self, is_255):
        self.is_255 = is_255

    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor

def mixup_batch(inp_batch, alpha):
    """
    Applies mixup augementation to a batch
    :param input_batch: tensor with batchsize as first dim
    :param alpha: lamda drawn from beta(alpha+1, alpha)
    """
    inp_clone = inp_batch.clone()
    #getting batch size
    batchsize = inp_batch.size()[0]

    #permute a clone
    perm = np.random.permutation(batchsize)
    for i in range(batchsize):
        inp_clone[i] = inp_batch[perm[i]]
    #generating different lambda for each sample
    #Refernced from http://www.inference.vc/mixup-data-dependent-data-augmentation/
    lam = torch.Tensor(np.random.beta(alpha+1, alpha, batchsize))
    lam = lam.view(-1,1,1,1)
    inp_mixup = lam * inp_batch + (1- lam) * inp_clone
    return inp_mixup

def onehot(t, num_classes):
    """
    convert index tensor into onehot tensor
    :param t: index tensor
    :param num_classes: number of classes
    """
    assert isinstance(t, torch.LongTensor)
    return torch.zeros(t.size()[0], num_classes).scatter_(1, t.view(-1, 1), 1)


def naive_cross_entropy_loss(output, target, size_average=True):
    """
    was used in mixup
    in PyTorch's cross entropy, targets are expected to be labels
    so to predict probabilities this loss is needed
    suppose q is the target and p is the input
    loss(p, q) = -\sum_i q_i \log p_i
    """
    if type(output) == tuple:
        output, _ = output
    if output.size() != target.size():
        print(output.size(), target.size())
        raise Exception("model not working")
    assert isinstance(output, Variable) and isinstance(target, Variable)
    output =-1 *  torch.log(Func.softmax(output).clamp(1e-5, 1))
    # input = input - torch.log(torch.sum(torch.exp(input), dim=1)).view(-1, 1)
    loss = torch.sum(output * target)

    return loss / output.size()[0] if size_average else loss
