import torch
import torch
import torch.nn as nn

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
        bsz, tree_len, d = trees.shape
        _, l, _ = lst.shape
        check(lst, [bsz, l, d])

        pos_one_hot = torch.zeros(1, tree_len, 1).to(trees.device).float()
        pos_one_hot[:, pos, :] = 1.
        pos_one_hot = pos_one_hot.to(trees.device)
        if low == high:
            lst_pos = lst[:, [low], :]
            lst_pos = lst_pos.repeat(1, tree_len, 1)
            check(lst_pos, [bsz, tree_len, d])
            trees = pos_one_hot * lst_pos + (1-pos_one_hot) * trees
            trees = self.rms_norm(trees)
            return trees

        mid = (low + high) // 2
        trees = self.build_neural_segment_trees(trees, lst, low=low, high=mid, pos=2*pos+1)
        trees = self.build_neural_segment_trees(trees, lst, low=mid + 1, high=high, pos=2*pos+2)

        in_pattern = torch.einsum("bi,bj->bij", trees[:, int(2*pos+1)], trees[:, int(2*pos+2)])
        alpha = self.rule(in_pattern.reshape(bsz, -1))
        alpha = torch.softmax(alpha, dim=-1)
        check(alpha, [bsz, 2])

        # convex mix of the identity operation applied to each argument
        source1 = alpha[:, [0]] * trees[:, int(2*pos+1)]
        source2 = alpha[:, [1]] * trees[:, int(2*pos+2)]
        mix = (source1 + source2).unsqueeze(1)
        check(mix, [bsz, 1, d])

        trees = pos_one_hot * mix.repeat(1, tree_len, 1) + (1-pos_one_hot) * trees
        trees = self.rms_norm(trees)
        check(trees, [bsz, tree_len, d])
        return trees

    def range_min_query(self, trees, query_low, query_high, low, high, pos, returns):
        """ perform range min queries in parallel """
        bsz, tree_len, d = trees.shape
        check(returns, [bsz, d])
        if pos >= tree_len:
            return returns

        # total overlap case
        mask_A = torch.logical_and(torch.less_equal(query_low, low), torch.greater_equal(query_high, high))
        mask_A = mask_A.reshape(bsz, 1).float()
        returns = mask_A * trees[:, pos, :] + (1 - mask_A) * returns
        check(returns, [bsz, d])

        # no overlap
        mask_B = torch.logical_or(torch.greater(query_low, high), torch.less(query_high, low))
        mask_B = mask_B.reshape(bsz, 1).float()
        inf_arg = self._get_inf_token(trees.device).reshape(1, -1).repeat([bsz, 1])
        check(inf_arg, [bsz, d])
        returns = mask_B * inf_arg + (1-mask_B) * returns
        check(returns, [bsz, d])

        # partial overlap
        mask_C = torch.logical_and(torch.logical_not(mask_A), torch.logical_not(mask_B)).float()
        check(mask_C, [bsz, 1])
        mid = ((low + high) // 2).long()
        arg1 = self.range_min_query(trees, query_low, query_high,
                                    low=low,
                                    high=mid,
                                    pos=2*pos+1,
                                    returns=returns)
        arg2 = self.range_min_query(trees, query_low, query_high,
                                    low=mid+1,
                                    high=high,
                                    pos=2*pos+2,
                                    returns=returns)
        check(arg1, [bsz, d])
        check(arg2, [bsz, d])
        in_pattern = torch.einsum("bi,bj->bij", arg1, arg2)
        alpha = self.rule(in_pattern.reshape(bsz, -1))
        check(alpha, [bsz, 2])
        alpha = torch.softmax(alpha, dim=-1)

        # convex mix of identity applied to each arg
        new_returns = alpha[:, [0]] * arg1 + alpha[:, [1]] * arg2
        check(new_returns, [bsz, d])

        returns = mask_C * new_returns + (1-mask_C) * returns
        returns = self.rms_norm(returns)
        return returns

    def forward(self, x, x_img, q, sharp_inference=False):
        # process input
        bsz, l = x.shape
        if self.use_images:
            check(x_img, [bsz, l, 28, 28])
            h = self.symbol_extractor(x_img.reshape(bsz * l, 28, 28)).reshape(bsz, l, -1)
        else:
            h = self.embedding(x)
            h = self.rms_norm(h)

        check(h, [bsz, l, self.symbol_repr_size])

        # create a new segment tree
        tree_len = lib.next_power_of_2(l) * 2 - 1
        inf = self.rms_norm(self._get_inf_token(x.device))
        trees = inf.repeat([bsz, tree_len, 1])

        # build segment trees up
        check(trees, [bsz, tree_len, self.symbol_repr_size])
        trees = self.build_neural_segment_trees(trees, h, low=0, high=l-1, pos=0)

        # prepare for query
        query_low = torch.tensor([t[0] for t in q]).long().to(x.device)
        query_high = torch.tensor([t[1] for t in q]).long().to(x.device)
        check(query_low, [bsz])
        check(query_high, [bsz])
        low = torch.zeros(bsz).long().to(x.device)
        high = torch.ones(bsz).mul(l-1).long().to(x.device)
        pos = 0
        returns = torch.zeros(bsz, self.symbol_repr_size).to(trees.device)

        # query segment tree
        returns = self.range_min_query(trees, query_low, query_high, low, high, pos, returns)

        logits = self.linear(returns)
        check(logits, [bsz, 10])

        return logits
