import os
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils import check
from torchvision import datasets, transforms


class Net(nn.Module):
    """
    Simple MNIST model.
    """
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def prepare_mnist_data():
    """ Download MNIST data and converts it into tensors. """
    if not os.path.exists("./DATA/MNIST"):
        print("Downloading MNIST ...")
        os.makedirs("./DATA")
        os.system("wget www.di.ens.fr/~lelarge/MNIST.tar.gz -P ./DATA")
        os.system("tar -zxvf ./DATA/MNIST.tar.gz --directory ./DATA")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root="./DATA",
                                   train=True,
                                   download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST(root="./DATA",
                                  train=False,
                                  download=True,
                                  transform=transform)

    return train_dataset.data, train_dataset.targets, test_dataset.data, test_dataset.targets


def get_conditional_sampler(x, y):
    """ Returns a function which allows randomly sampling images conditioned on the class. """
    x_split = [x[y == i].float() for i in range(10)]

    def sampler(digit):
        assert 0 <= digit <= 9, f"{digit} is not 0-9"
        idx = random.randint(0, x_split[digit].shape[0]-1)
        return x_split[digit][idx]
    return sampler


def next_power_of_2(number):
    """ Returns the next power of 2. """
    if number <= 0:
        return 1

    bit_len = int.bit_length(number)
    power2 = 2 ** (bit_len - 1)

    if power2 == number:
        return number
    else:
        return power2 * 2


def create_segment_tree(lst):
    """ Creates segment tree from input list. """
    tree_len = next_power_of_2(len(lst)) * 2 - 1
    tree = [math.inf] * tree_len
    build_min_seg_tree(tree, lst, low=0, high=len(lst) - 1, pos=0)

    return tree


def build_min_seg_tree(tree, lst, low, high, pos):
    """ Recursive function to build segment tree bottom up. """
    if low == high:
        tree[pos] = lst[low]
        # print("tree: ", tree)
        return
    mid = (low + high) // 2
    build_min_seg_tree(tree, lst, low=low, high=mid, pos=2 * pos + 1)
    build_min_seg_tree(tree, lst, low=mid + 1, high=high, pos=2 * pos + 2)
    # print("left_child: ", tree[2 * pos + 1])
    # print("right_child: ", tree[2 * pos + 2])
    tree[pos] = min(tree[int(2 * pos + 1)], tree[int(2 * pos + 2)])
    # print("tree: ", tree)


def range_min_query(tree, query_low, query_high, low, high, pos):
    """
    Query range on list input using its segment tree.

    tree: segment tree of list input
    low: range begin of input list
    high: range end of input list
    query_low: query range begin in input list
    query_high: query range end in input list
    pos: current pointer location
    """
    # total overlap case
    if query_low <= low and query_high >= high:
        return tree[pos]

    # no overlap
    if query_low > high or query_high < low:
        return math.inf

    # partial overlap
    mid = (low + high) // 2
    arg1 = range_min_query(tree, query_low, query_high,
                           low=low,
                           high=mid,
                           pos=2 * pos + 1)
    arg2 = range_min_query(tree, query_low, query_high,
                           low=mid + 1,
                           high=high,
                           pos=2 * pos + 2)
    # print(f"compare {arg1} and {arg2}")
    return min(arg1, arg2)

