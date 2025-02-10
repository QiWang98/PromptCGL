# -*- coding: utf-8 -*-
# @Time    : 2024/5/14
# @Author  : Qi Wang
import random

import torch
import numpy as np
from dgllife.utils.splitters import RandomSplitter
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
from torch_sparse import SparseTensor
from progressbar import progressbar


def convert_dgl_to_pyg(dgl_graph, label):
    edge_index = dgl_graph[1].edges()
    edge_index = torch.stack(edge_index).long()
    adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                       sparse_sizes=(dgl_graph[1].num_nodes(), dgl_graph[1].num_nodes()))
    x = dgl_graph[1].ndata['h']
    data = Data(x=x, edge_index=edge_index, adj_t=adj.t().to_symmetric(), y=label)
    return data


class Streaming:
    def __init__(self, args, dataset, task_type="node_cls"):
        self.args = args
        self.cls_per_task = args.cls_per_task
        self.dataset_name = args.dataset_name
        self.task_type = task_type
        if self.dataset_name in ["corafull", "arxiv", "Reddit", "products"]:
            self.tasks = self.prepare_node_cls_tasks(dataset)
            self.n_tasks = len(self.tasks)

    def prepare_node_cls_tasks(self, dataset):
        graph = dataset[0]
        tasks = []
        n_tasks = dataset.num_classes // self.cls_per_task
        for k in progressbar(range(n_tasks), redirect_stdout=True):
            start_cls = k * self.cls_per_task
            classes = list(range(start_cls, start_cls + self.cls_per_task))
            subset = sum(graph.y == cls for cls in classes).nonzero(as_tuple=False).squeeze()
            subgraph = graph.subgraph(subset)

            # Split to train/val/test
            transform = RandomNodeSplit(num_val=0.2, num_test=0.2)
            subgraph = transform(subgraph)

            edge_index = subgraph.edge_index
            adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                               sparse_sizes=(subgraph.num_nodes, subgraph.num_nodes))
            subgraph.adj_t = adj.t().to_symmetric()  # Arxiv is a directed graph.

            subgraph.task_id = k
            subgraph.classes = classes

            tasks.append(subgraph)
        return tasks

