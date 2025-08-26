<h1 align="center">
Prompt-Driven Continual Graph Learning
</h1>

[Qi Wang](), [Tianfei Zhou](https://www.tfzhou.com/), [Ye Yuan](), [Rui Mao]()

----------

This is the official repository for the paper [_Prompt-Driven Continual Graph Learning_](https://arxiv.org/abs/2502.06327).

The following figure illustrates the overall PromptCGL framework, which mitigates catastrophic forgetting via hierarchical prompting and personalized prompt generation.

<div align="center">
  <img src="arch.png" alt="Framework" width="80%" height="80%">
</div>
<br>

----

# Experiment Environment
Our experiments are conducted on Python 3.10 and CUDA 12.2, with the following package dependencies:

```
torch==1.13.1            # core deep learning framework
dgl==1.1.3               # graph neural network library
torch-geometric==2.4.0   # PyG for deploying GNNs
ogb==1.3.6               # for Arxiv and Products datasets
dgllife==0.3.2           # graph datasets support
progressbar2             # visualization of training progress
```

----

# Usage
To run training and evaluation on CoraFull with PromptCGL:
```
python main.py --dataset-name=corafull --cgl-method=PromptCGL
```
----

# Cite
If you find this repository useful, please cite:
```
@article{wang2025prompt,
  title={Prompt-Driven Continual Graph Learning},
  author={Wang, Qi and Zhou, Tianfei and Yuan, Ye and Mao, Rui},
  journal={arXiv preprint arXiv:2502.06327},
  year={2025}
}
```

----

# Credit
This repository was developed based on the [CGLB](https://github.com/QueuQ/CGLB).
