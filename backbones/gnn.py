# -*- coding: utf-8 -*-
# @Time    : 2024/5/6
# @Author  : Qi Wang
import torch
import torch.nn.functional as F

class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([])

    def initialize(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data, prompt=None):
        x, adj_t = data.x, data.adj_t
        for i, layer in enumerate(self.layers[:-1]):
            if prompt is not None:
                x = prompt.add(x, i)
            x = layer(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        if prompt is not None:
            x = prompt.add(x, len(self.layers) - 1)
        x = self.layers[-1](x, adj_t)
        return x

    def encode(self, x, adj_t):
        self.eval()
        for i, layer in enumerate(self.layers[:-1]):  # without the FC layer.
            x = layer(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        return x

    def encode_noise(self, x, adj_t):
        self.eval()
        for i, layer in enumerate(self.layers):
            x = layer(x, adj_t)
            if i != len(self.layers) - 1:
                x = F.relu(x)
        random_noise = torch.rand_like(x).cuda()
        x += torch.sign(x) * F.normalize(random_noise, dim=-1) * 0.1
        return x


def train_node_classifier(model, data, optimizer, weight=None, n_epoch=200, incremental_cls=None):
    model.train()
    ce = torch.nn.CrossEntropyLoss(weight=weight)
    for epoch in range(n_epoch):
        if incremental_cls:
            out = model(data)[:, 0:incremental_cls[1]]
        else:
            out = model(data)

        loss = ce(out[data.train_mask], data.y[data.train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def train_node_classifier_batch(model, batches, optimizer, n_epoch=200, incremental_cls=None):
    model.train()
    ce = torch.nn.CrossEntropyLoss()
    for _ in range(n_epoch):
        for data in batches:
            if incremental_cls:
                out = model(data)[:, 0:incremental_cls[1]]
            else:
                out = model(data)

            loss = ce(out[data.train_mask], data.y[data.train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def eval_node_classifier(model, data, incremental_cls=None):
    model.eval()
    with torch.no_grad():
        pred = model(data)[data.test_mask, incremental_cls[0]:incremental_cls[1]].argmax(dim=1)
        correct = pred.eq(data.y[data.test_mask] - incremental_cls[0]).sum()
        acc = int(correct) / int(data.test_mask.sum())
    return acc
