import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

import lib
from lib import Net
from utils import check


class Model(nn.Module):
    def __init__(self,
                 use_images=True,
                 symbol_repr_size=15,
                 n_classes=10):
        super().__init__()
        self.use_images = use_images
        self.inf_idx = n_classes
        self.symbol_repr_size = symbol_repr_size
        self.symbol_extractor = Net(self.symbol_repr_size)
        self.rms_norm = lambda x: x / x.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt().add(1e-6)

        self.embedding = nn.Embedding(n_classes+1, self.symbol_repr_size)  # +1 for inf token
        self.linear = nn.Linear(self.symbol_repr_size, n_classes, bias=False)

        self.rule = nn.Linear(self.symbol_repr_size ** 2, 2, bias=False)

    def _get_inf_token(self, device):
        inf_id = torch.tensor([[self.inf_idx]]).long().to(device)
        return self.embedding(inf_id)

    def build_neural_segment_trees(self, trees, lst, low, high, pos):
        """ Recursive function to build a segment tree for each batch element. """
        print("depth")
        print("low", low)
        print("high", high)
        print("pos", pos)
        print()
        bsz, tree_len, d = trees.shape
        if low == high:
            trees[:, pos] = lst[:, low]
            return

        mid = (low + high) // 2
        self.build_neural_segment_trees(trees, lst, low=low, high=mid, pos=2*pos+1)
        self.build_neural_segment_trees(trees, lst, low=mid + 1, high=high, pos=2*pos+2)

        in_pattern = torch.einsum("bi,bj->bij", trees[:, int(2*pos+1)], trees[:, int(2*pos+2)])
        alpha = self.rule(in_pattern.reshape(bsz, -1))
        alpha = torch.softmax(alpha, dim=-1)
        check(alpha, [bsz, 2])

        # convex mix of the identity operation applied to each argument
        source1 = alpha[:, [0]] * trees[:, int(2*pos+1)]
        source2 = alpha[:, [1]] * trees[:, int(2*pos+2)]
        check(source1, [bsz, d])
        trees[:, pos] = self.rms_norm(source1 + source2)
        check(trees, [bsz, tree_len, d])
        return trees

    def range_min_query(self, trees, query_low, query_high, low, high, pos, returns):
        bsz, l, d = trees.shape
        check(returns, [bsz, d])

        # total overlap case
        mask = torch.logical_and(query_low <= low, query_high >= high)
        if mask.any():
            print("mask.shape: ", mask.shape)
            print("pos.shape: ", pos.shape)
            print("pos: ", pos)
            print("returns[mask].shape", returns[mask].shape)
            print("trees[mask][pos].shape", trees[mask][pos].shape)

            returns[mask] = trees[mask][pos]

        # no overlap
        mask = torch.logical_or(query_low > high, query_high < low)
        inf_arg = self._get_inf_token(trees.device).reshape(1, -1).repeat([bsz, 1])
        returns[mask] = inf_arg[mask]

        # partial overlap
        mid = ((low + high) // 2).long()
        arg1 = self.range_min_query(trees, query_low, query_high,
                                    low=low,
                                    high=mid,
                                    pos=pos.mul(2).add(1).long(),
                                    returns=returns)
        arg2 = self.range_min_query(trees, query_low, query_high,
                                    low=mid+1,
                                    high=high,
                                    pos=pos.mul(2).add(2).long(),
                                    returns=returns)
        check(arg1, [bsz, d])
        check(arg2, [bsz, d])
        in_pattern = torch.einsum("bi,bj->bij", arg1, arg2)
        alpha = self.rule(in_pattern.reshape(bsz, -1))
        check(alpha, [bsz, 2])
        alpha = torch.softmax(alpha, dim=-1)

        # convex mix of identity applied to each arg
        returns = self.rms_norm(alpha[:, 0] * arg1 + alpha[:, 1] * arg2)

        return returns,

    def forward(self, x, x_img, q, sharp_inference=False):
        # process input
        bsz, l = x.shape
        if self.use_images:
            check(x_img, [bsz, l, 28, 28])
            h = self.net(x_img.reshape(bsz * l, 28, 28)).reshape(bsz, l, -1)
        else:
            h = self.embedding(x)
            h = self.rms_norm(h)

        check(h, [bsz, l, self.symbol_repr_size])

        # create a segment tree
        tree_len = lib.next_power_of_2(l)*2-1
        inf = self._get_inf_token(x.device)
        trees = inf.repeat([bsz, tree_len, 1])


        # build segment trees
        print(" ---- build segment trees")
        check(trees, [bsz, tree_len, self.symbol_repr_size])
        trees = self.build_neural_segment_trees(trees, h, low=0, high=l-1, pos=0)

        # prepare for query
        print(" ---- range min query")
        query_low = torch.tensor([t[0] for t in q]).long().to(x.device)
        query_high = torch.tensor([t[1] for t in q]).long().to(x.device)
        check(query_low, [bsz])
        check(query_high, [bsz])

        low = torch.zeros(bsz).long().to(x.device)
        high = torch.ones(bsz).mul(l).long().to(x.device)
        pos = torch.zeros(bsz).long().to(x.device)
        returns = torch.zeros(bsz, self.symbol_repr_size)
        y_hat = self.range_min_query(trees, query_low, query_high, low, high, pos, returns)

        # call call call
        for _ in range(10):
            returns = self.range_min_query(trees, query_low, query_high, low, high, pos, returns)

        logits = self.linear(returns)
        check(logits, [bsz, 10])
        return logits
