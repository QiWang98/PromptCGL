# -*- coding: utf-8 -*-
# @Time    : 2024/5/13
# @Author  : Qi Wang
import os.path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch


class AttPrompt(nn.Module):
    def __init__(self, nin, nhid, num_layer, p_num=3):
        super(AttPrompt, self).__init__()
        self.p_list = torch.nn.ParameterList()
        self.a_list = torch.nn.ParameterList()
        self.p_list.append(nn.Parameter(torch.Tensor(p_num, nin)))  # before input layer
        self.a_list.append(nn.Parameter(torch.Tensor(nin, 1)))
        self.a_list.append(nn.Parameter(torch.Tensor(1, p_num)))

        for layer in range(num_layer - 1):
            self.p_list.append(nn.Parameter(torch.Tensor(p_num, nhid)))
            self.a_list.append(nn.Parameter(torch.Tensor(nhid, 1)))
            self.a_list.append(nn.Parameter(torch.Tensor(1, p_num)))
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.p_list:
            nn.init.xavier_uniform_(p)
        for a in self.a_list:
            nn.init.xavier_uniform_(a)

    def add(self, x: torch.Tensor, layer: int = 0):
        a_matrix = self.a_list[layer*2].mm(self.a_list[layer*2+1])
        score = x.mm(a_matrix)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list[layer])

        return x + p


class PromptCGL():
    def __init__(self, model, tasks, device, args):
        super().__init__()
        self.model = model
        self.tasks = tasks
        self.device = device
        self.ce = torch.nn.CrossEntropyLoss()
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.args = args

    def observer(self, epoches, IL):
        tasks = self.tasks
        performance_matrix = torch.zeros(len(tasks) + 1, len(tasks) + 1)
        APs = []
        for k in range(len(tasks)):
            task = tasks[k]
            task.to(self.device, "x", "y", "adj_t")
            num_cls = torch.unique(task.y)[-1]

            # Train
            if k > 0:
                prompt = AttPrompt(self.model.layers[0].in_channels, self.model.layers[0].out_channels,
                                   len(self.model.layers), p_num=self.args['p_num']).to(self.device)
                optimizer = torch.optim.Adam([{"params": self.model.layers[-1].parameters(),
                                               "lr": self.args['lr_tuning'], "weight_decay": self.args['wd_tuning']},
                                              {"params": prompt.parameters(),
                                               "lr": self.args['lr_prompt'], "weight_decay": self.args['wd_prompt']}])
            else:
                prompt = None
                optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4)

            for _ in range(epoches):
                output = self.model(task, prompt)[:, :num_cls + 1]
                loss = self.ce(output[task.train_mask], task.y[task.train_mask])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if k > 0:
                path_dir = os.path.join("./models", "prompts")
                if not os.path.exists(path_dir):
                    os.makedirs(path_dir)
                torch.save(prompt.state_dict(), os.path.join(path_dir, f"prompt_{k}.pt"))

            # Evaluation
            accs = []
            AF = 0
            for k_ in range(k + 1):
                task_ = tasks[k_].to(self.device, "x", "y", "adj_t")
                self.model.eval()
                if k_ > 0:
                    prompt = AttPrompt(self.model.layers[0].in_channels, self.model.layers[0].out_channels,
                                       len(self.model.layers), p_num=self.args['p_num']).to(self.device)
                    prompt.load_state_dict(torch.load(f"./models/prompts/prompt_{k_}.pt"))
                    prompt.eval()
                else:
                    prompt = None
                if IL == "classIL":
                    pred = self.model(task_, prompt)[task_.test_mask, :num_cls + 1].argmax(dim=1)
                    correct = pred.eq(task_.y[task_.test_mask]).sum()
                    acc = (int(correct) / int(task_.test_mask.sum())) * 100
                else:
                    num_cls = torch.unique(task_.y)[-1]
                    min_cls = torch.unique(task_.y)[0]
                    pred = self.model(task_, prompt)[task_.test_mask, min_cls: num_cls+1].argmax(dim=1)
                    correct = pred.eq(task_.y[task_.test_mask] - min_cls).sum()
                    acc = (int(correct) / int(task_.test_mask.sum())) * 100
                accs.append(acc)
                task_.to("cpu")
                print(f"T{k_} {acc:.2f}", end="|", flush=True)
                performance_matrix[k, k_] = acc
            AP = sum(accs) / len(accs)
            APs.append(AP)
            print(f"AP: {AP:.2f}", end=", ", flush=True)
            for t in range(k):
                AF += performance_matrix[k, t] - performance_matrix[t, t]
            AF = AF / k if k != 0 else AF
            print(f"AF: {AF:.2f}", flush=True)
        return AP, np.mean(APs), AF, performance_matrix

    def get_embedding(self, class_id, dataset_name, is_prompt=False):

        tasks = self.tasks
        task_id = class_id // 2

        data = tasks[task_id]

        mask = data.y == class_id
        data.to(self.device, "x", "y", "adj_t")
        if is_prompt:
            prompt = AttPrompt(self.model.layers[0].in_channels, self.model.layers[0].out_channels,
                               len(self.model.layers), p_num=self.args['p_num']).to(self.device)

            self.model.eval()
            with torch.no_grad():
                if task_id > 0:
                    prompt_path = os.path.join("./models", dataset_name, "prompts", "prompt_{}.pt".format(task_id))
                    prompt.load_state_dict(torch.load(prompt_path))
                    prompt.eval()
                    output = self.model(data, prompt)
                else:
                    output = self.model(data)
        else:
            self.model.eval()
            with torch.no_grad():
                output = self.model(data)
        output = output[mask]

        return output.cpu().numpy()
