import torch
import torch.nn.functional as F
from backbones.gnn import GNN
from torch_sparse import SparseTensor
from torch_geometric.nn import GATConv, global_mean_pool


class GAT(GNN):
    def __init__(self, nin, nhid, nout, nlayers, concat=False, heads=8):
        super().__init__()
        if nlayers == 1:
            self.layers.append(GATConv(nin, nout, concat=concat, heads=1))
        else:
            self.layers.append(GATConv(nin, nhid, concat=concat, heads=heads))  # input layers
            for _ in range(nlayers - 2):
                self.layers.append(GATConv(nhid, nhid, concat=concat, heads=heads))  # hidden layers
            self.layers.append(GATConv(nhid, nout, concat=concat, heads=1))  # output layers
