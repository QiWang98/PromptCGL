# -*- coding: utf-8 -*-
# @Time    : 2024/5/6
# @Author  : Qi Wang
import torch
import torch.nn.functional as F
from backbones.gnn import GNN
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN(GNN):
    def __init__(self, nin, nhid, nout, nlayers):
        super().__init__()
        if nlayers == 1:
            self.layers.append(GCNConv(nin, nout))
        else:
            self.layers.append(GCNConv(nin, nhid))  # input layers
            for _ in range(nlayers - 2):
                self.layers.append(GCNConv(nhid, nhid))  # hidden layers
            self.layers.append(GCNConv(nhid, nout))  # output layers