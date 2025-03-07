# -*- coding: utf-8 -*-
# @Time    : 2024/5/13
# @Author  : Qi Wang
import torch
from torch_geometric import seed_everything

# Utility
import time
import argparse
import numpy as np
from utilities import *
from data_stream import Streaming
from torch_geometric.data import Batch
from backbones.gnn import train_node_classifier, train_node_classifier_batch, eval_node_classifier


def evaluate(args, dataset, data_stream, memory_banks, flush=True):
    APs, AFs, mAPs = [], [], []
    Ps = []
    for i in range(args.repeat):
        memory_bank = memory_banks[i]
        # Initialize the performance matrix.
        performace_matrix = torch.zeros(len(memory_bank), len(memory_bank))
        model = get_backbone_model(dataset, data_stream, args)
        cgl_model = get_cgl_model(model, data_stream, args)
        tasks = cgl_model.tasks

        opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        mAP = 0
        for k in range(len(memory_bank)):
            # train
            if args.dataset_name == "products" and args.cgl_method == "joint":
                max_cls = torch.unique(memory_bank[k].y)[-1]
                batches = memory_bank[:k + 1]
                for data in batches:
                    data.to(args.device)
                model = train_node_classifier_batch(model, batches, opt, n_epoch=args.cls_epoch,
                                                    incremental_cls=(0, max_cls + 1))
            else:
                if args.tim:
                    if args.batch:
                        replayed_graphs = memory_bank[:k + 1]
                    else:
                        replayed_graphs = Batch.from_data_list(memory_bank[:k + 1])
                else:
                    if args.batch:
                        replayed_graphs = memory_bank[:k] + [tasks[k]]
                    else:
                        replayed_graphs = Batch.from_data_list(memory_bank[:k] + [tasks[k]])

                if args.batch:
                    max_cls = torch.unique(memory_bank[k].y)[-1]
                    batches = replayed_graphs
                    for data in batches:
                        data.to(args.device)
                    model = train_node_classifier_batch(model, batches, opt, n_epoch=args.cls_epoch,
                                                        incremental_cls=(0, max_cls + 1))
                else:
                    replayed_graphs.to(args.device, "x", "y", "adj_t")
                    max_cls = torch.unique(replayed_graphs.y)[-1]
                    model = train_node_classifier(model, replayed_graphs, opt, weight=None, n_epoch=args.cls_epoch,
                                                  incremental_cls=(0, max_cls + 1))

            # Test the model from task 0 to task k
            accs = []
            AF = 0
            for k_ in range(k + 1):
                task_ = tasks[k_].to(args.device, "x", "y", "adj_t")
                if args.IL == "classIL":
                    acc = eval_node_classifier(model, task_, incremental_cls=(0, max_cls + 1)) * 100
                else:
                    max_cls = torch.unique(task_.y)[-1]
                    acc = eval_node_classifier(model, task_, incremental_cls=(max_cls + 1 - data_stream.cls_per_task,
                                                                              max_cls + 1)) * 100
                accs.append(acc)
                task_.to("cpu")
                print(f"T{k_} {acc:.2f}", end="|", flush=flush)
                performace_matrix[k, k_] = acc
            AP = sum(accs) / len(accs)
            mAP += AP
            print(f"AP: {AP:.2f}", end=", ", flush=flush)
            for t in range(k):
                AF += performace_matrix[k, t] - performace_matrix[t, t]
            AF = AF / k if k != 0 else AF
            print(f"AF: {AF:.2f}", flush=flush)
        APs.append(AP)
        AFs.append(AF)
        mAPs.append(mAP / (k + 1))
        Ps.append(performace_matrix)
    print(f"AP: {np.mean(APs):.1f}±{np.std(APs, ddof=1):.1f}", flush=flush)
    print(f"mAP: {np.mean(mAPs):.1f}±{np.std(mAPs, ddof=1):.1f}", flush=flush)
    print(f"AF: {np.mean(AFs):.1f}±{np.std(AFs, ddof=1):.1f}", flush=flush)
    return Ps


def main():
    parser = argparse.ArgumentParser()
    # Arguments for data.n
    parser.add_argument('--dataset-name', type=str, default="corafull")
    # corafull, arxiv, Reddit, products
    parser.add_argument('--cls-per-task', type=int, default=2)
    parser.add_argument('--data-dir', type=str, default="./data")
    parser.add_argument('--result-path', type=str, default="./results")

    # Argumnets for CGL methods.
    parser.add_argument('--tim', action='store_true')
    parser.add_argument('--cgl-method', type=str, default="PromptCGL")  # PromptCGL
    parser.add_argument('--cls-epoch', type=int, default=200)
    # Corafull 8710, arxiv 128, Reddit 602, products 100
    parser.add_argument('--budget', type=int, default=2)
    parser.add_argument('--m-update', type=str, default="all")
    parser.add_argument('--cgm-args', type=str, default="{'n_encoders': 3000, 'feat_init': 'randomChoice', 'feat_lr': 1e-3, 'hid_dim': 512, 'emb_dim': 256, 'n_layers': 2, 'hop': 1, 'activation': True}")
    parser.add_argument('--ewc-args', type=str, default="{'memory_strength': 100000.}")
    parser.add_argument('--mas-args', type=str, default="{'memory_strength': 10000.}")
    parser.add_argument('--gem-args', type=str, default="{'memory_strength': 0.5, 'n_memories': 20}")
    parser.add_argument('--twp-args', type=str, default="{'lambda_l': 10000., 'lambda_t': 10000., 'beta': 0.01}")
    parser.add_argument('--lwf-args', type=str, default="{'lambda_dist': 10., 'T': 20.}")
    parser.add_argument('--promptcgl_args', type=str, default="{'p_num': 3, 'lr_tuning': 5e-4, 'wd_tuning': 0, "
                                                          "'lr_prompt': 1e-2, 'wd_prompt': 5e-4}")  # arxiv: 'lr_prompt': 5e-2'
    parser.add_argument('--IL', type=str, default="taskIl")
    parser.add_argument('--batch', action='store_true')

    # Others
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1024)  # Set the seed for comparison.
    parser.add_argument('--rewrite', action='store_true')
    args = parser.parse_args()
    if args.dataset_name in ["SIDER", "Tox21"]:
        args.cls_per_task = 1

    seed_everything(args.seed)
    print(args)

    # Get file names.
    if args.cgl_method in ["bare", "ewc", "mas", "gem", "twp", "lwf", "PromptCGL"]:
        result_file_name = f"{args.dataset_name}_{args.cgl_method}"
    else:
        result_file_name = get_result_file_name(args)

    memory_bank_file_name = os.path.join(args.result_path, "memory_bank", result_file_name)
    task_file = os.path.join(args.data_dir, "streaming", f"{args.dataset_name}.streaming")

    dataset = get_dataset(args)
    if os.path.exists(task_file):
        data_stream = torch.load(task_file)
    else:
        data_stream = Streaming(args, dataset)
        torch.save(data_stream, task_file)
    # corafull, arxiv, Reddit, products
    if args.cgl_method in ["bare", "ewc", "mas", "gem", "twp", "lwf", "PromptCGL"]:
        APs, AFs, mAPs = [], [], []
        Ps = []
        for i in range(args.repeat):
            model = get_backbone_model(dataset, data_stream, args)
            cgl_model = get_cgl_model(model, data_stream, args)
            AP, mAP, AF, performace_matrix = cgl_model.observer(args.cls_epoch, args.IL)
            APs.append(AP)
            AFs.append(AF)
            mAPs.append(mAP)
            Ps.append(performace_matrix)
        print(f"AP: {np.mean(APs):.1f}±{np.std(APs, ddof=1):.1f}", flush=True)
        print(f"mAP: {np.mean(mAPs):.1f}±{np.std(mAPs, ddof=1):.1f}", flush=True)
        print(f"AF: {np.mean(AFs):.1f}±{np.std(AFs, ddof=1):.1f}", flush=True)
        torch.save(Ps, os.path.join(args.result_path, "performance_new", f"{result_file_name}.pt"))
        # 保持cgl_model
        torch.save(cgl_model, os.path.join("./models", args.cgl_method + "_" + args.dataset_name + ".pt"))
    else:
        # Get memory banks.
        memory_banks = []
        for i in range(args.repeat):
            if os.path.exists(memory_bank_file_name + f"_repeat_{i}") and not args.rewrite:
                memory_bank = torch.load(memory_bank_file_name + f"_repeat_{i}")
                memory_banks.append(memory_bank)  # load the memory bank from the file.
            else:
                model = get_backbone_model(dataset, data_stream, args)
                cgl_model = get_cgl_model(model, data_stream, args)

                memory_bank = cgl_model.observer()
                memory_banks.append(memory_bank)
                torch.save(memory_bank, memory_bank_file_name + f"_repeat_{i}")

        Ps = evaluate(args, dataset, data_stream, memory_banks)

        if args.tim:
            if args.batch:
                torch.save(Ps, os.path.join(args.result_path, "performance_new", f"{result_file_name}_tim_batch.pt"))
            else:
                torch.save(Ps, os.path.join(args.result_path, "performance_new", f"{result_file_name}_tim.pt"))
        else:
            if args.batch:
                torch.save(Ps, os.path.join(args.result_path, "performance_new", f"{result_file_name}_batch.pt"))
            else:
                torch.save(Ps, os.path.join(args.result_path, "performance_new", f"{result_file_name}.pt"))


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    start_time = time.time()
    main()
    print(f"Time: {(time.time() - start_time) / 60:.2f} min")
