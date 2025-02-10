"""
@author: Qi Wang
@data: 2024/7/15
"""
import torch
import torch.nn.functional as F
from backbones.gnn import GNN
from torch_geometric.nn import SAGEConv

class SAGE(GNN):
    def __init__(self, nin, nhid, nout, nlayers):
        super().__init__()
        if nlayers == 1:
            self.layers.append(SAGEConv(nin, nout))
        else:
            self.layers.append(SAGEConv(nin, nhid))  # input layers
            for _ in range(nlayers - 2):
                self.layers.append(SAGEConv(nhid, nhid))  # hidden layers
            self.layers.append(SAGEConv(nhid, nout))  # output layers