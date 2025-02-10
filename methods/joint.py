# -*- coding: utf-8 -*-
# @Time    : 2024/5/18
# @Author  : Qi Wang
import torch
import numpy as np
from torch_geometric.data import Batch
from methods.replay import Replay


class Joint(Replay):
    def __init__(self, model, tasks, budget, m_update, device):
        super().__init__(model, tasks, budget, m_update, device)

    def memorize(self, task, budgets):
        return task
