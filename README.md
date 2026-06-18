## ACS

This is the source code for our paper: **Dynamic Deep Neural Network Inference via Adaptive Channel Skipping**. A brief introduction of this work is as follows:

> Deep Neural Networks have recently made remarkable achievements in computer vision applications. However, the high computational requirements needed to achieve accurate inference results can be a significant barrier to deploying DNNs on resource-constrained computing devices, such as those found in the Internet-of-Things. In this work, we propose a fresh approach called Adaptive Channel Skipping (ACS) that prioritizes the identification of the most suitable channels for skipping and implements an efficient skipping mechanism during inference. We begin with the development of a new Gating Network model, ACS-GN, which employs fine-grained channel-wise skipping to enable input-dependent inference and achieve a desirable balance between accuracy and resource consumption. To further enhance the efficiency of channel skipping, we propose a Dynamic Grouping convolutional computing approach, ACS-DG, which helps to reduce the computational cost of ACS-GN. The results of our experiment indicate that ACS-GN and ACS-DG exhibit superior performance compared to existing gating network designs and convolutional computing mechanisms, respectively. When they are combined, the ACS framework results in a significant reduction of computational expenses and a remarkable improvement in the accuracy of inferences.

> 深度神经网络近期在计算机视觉应用领域取得了显著成就。然而，要实现精确推理结果所需的高计算量，可能成为在资源受限的计算设备（如物联网设备）上部署深度神经网络的主要障碍。本研究提出了一种名为自适应通道跳跃（ACS）的创新方法，该方法优先识别最适合跳过的通道，并在推理过程中实现高效跳跃机制。我们首先开发了新型门控网络模型ACS-GN，该模型采用细粒度通道级跳跃技术，既能实现输入依赖型推理，又能在精度与资源消耗之间达成理想平衡。为进一步提升通道跳跃效率，我们提出了动态分组卷积计算方法ACS-DG，有效降低了ACS-GN的计算成本。实验结果表明：ACS-GN在门控网络设计方面、ACS-DG在卷积计算机制方面，分别优于现有方案。当两者结合时，ACS框架能显著降低计算开销，并大幅提升推理精度。

## Required software

- PyTorch
- torchvision
- NumPy

## Project Structure

```
ACS/
├── dgconv.py                              # Core module: Dynamic Grouping Convolution (ACS-DG)
├── g_resnext.py                           # ResNeXt model built on DGConv2d
├── models_channel_skip_new_gate.py        # DenseNet + Gate Network (ACS-GN)
├── train_base_channel_skip_new_gate.py    # Training / evaluation entry script
└── README.md
```

## Core Modules

### Dynamic Grouping Convolution (`dgconv.py`)

Implements `DGConv2d`, the core operator of the **ACS-DG** approach. It learns a compact binary gate (of length `K = log2(in_channels)`) and expands it into a channel-wise mask `U` via Kronecker-product aggregation, so that the convolution weight is dynamically masked at each forward pass.

**Key components:**

- `aggregate(gate, D, I, K, sort)` — Aggregates `K` binary gates into a full channel mask `U` through iterated Kronecker products. `D` is the 2×2 identity (skip) and `I` is the 2×2 all-ones (keep) matrix.
- `kronecker_product(mat1, mat2)` — Helper for the Kronecker product used in the aggregation.
- `DGConv2d` — Wraps a standard `nn.Conv2d` and applies the learned mask `U` to its weight. A **Straight-Through Estimator (STE)** is used so the binary gate remains differentiable.

**Key behavior:**

- `forward(x)` returns `(output, U_regularizer)`, where `U_regularizer = 2^(K + sum(gate))` penalizes the number of retained channels and encourages sparsity.
- The gate is initialized with a tiny random `±eps` value so the network starts near the decision boundary.

### ResNeXt Backbone (`g_resnext.py`)

A ResNeXt implementation that replaces 3×3 convolutions with `DGConv2d` while keeping 1×1 convolutions standard. It accumulates the `U_regularizer` term from every DGConv2d layer and exposes it alongside the classification logits.

**Key components:**

- `conv3x3` — 3×3 `DGConv2d` with padding/dilation support.
- `conv1x1` — Standard 1×1 `nn.Conv2d`.
- `Bottleneck` — 3-layer bottleneck (1×1 → 3×3 DGConv → 1×1), accumulates `U_regularizer` across blocks.
- `G_ResNet` — Standard ResNet trunk (`conv1` → 4 `layerX` → `avgpool` → `fc`); `forward` returns `(logits, U_regularizer_sum)` summed over all four stages.
- `g_resnext50` / `g_resnext101` — Factory functions following the ResNeXt-50 32×4d / ResNeXt-101 32×8d configurations.

### DenseNet with Gate Network (`models_channel_skip_new_gate.py`)

Implements the **ACS-GN** approach: a DenseNet backbone where each dense layer is followed by a `NewGate` that produces an input-dependent binary channel mask, enabling fine-grained channel-wise skipping.

**Key components:**

- `DGConv2d` / `aggregate` / `kronecker_product` — A self-contained copy of the ACS-DG operator (note: the STE here uses `(>0).float().detach() - data.detach() + data`, slightly different from `dgconv.py`).
- `SELayer` — Squeeze-and-Excitation channel attention (defined but not enabled in the default `NewGate`).
- `NewGate` — The gate network: `1×1 conv → 3×3 DGConv → BN → ReLU → 3×3 DGConv → avgpool → 1×1 conv → sigmoid → discretization`. Returns `(mask, logprob)`.
- `BasicBlock` / `_DenseLayer` / `_Transition` / `DenseNet` — DenseNet building blocks; each dense layer output is multiplied by the gate mask before being concatenated.
- `densenet40` — Factory function with `block_config=(6, 6, 6)`.

**Forward output:** `DenseNet.forward(x)` returns `(output, masks, gprobs)`, where `masks` is the list of per-layer binary channel masks and `gprobs` is the list of per-layer log-probabilities used for regularization.

### Training Script (`train_base_channel_skip_new_gate.py`)

Entry point for training and evaluating the DenseNet + gate-network model on CIFAR-10/100 or SVHN.

**Key components:**

- `parse_args()` — Command-line arguments (see table below).
- `prepare_train_data` / `prepare_test_data` — Dataset loaders for CIFAR-10/100 and SVHN.
- `run_training(args)` — Builds the model, optionally resumes from a checkpoint, runs SGD with step-decay learning rate, and saves checkpoints per epoch.
- `train(...)` / `validate(...)` — Standard training / validation loops logging Top-1/Top-5 accuracy.
- `save_checkpoint(...)` / `adjust_learning_rate(...)` / `accuracy(...)` / `AverageMeter` — Utilities.

**Key command-line arguments:**

| Argument | Default | Description |
|---|---|---|
| `cmd` | — | `train` / `test` / `map` / `locate` |
| `arch` | `resnet74` | Model factory name in `models_channel_skip_new_gate` (e.g. `densenet40`) |
| `-d, --dataset` | `cifar10` | Dataset: `cifar10` / `cifar100` |
| `--epochs` | `120` | Total number of training epochs |
| `-b, --batch-size` | `128` | Mini-batch size |
| `--lr` | `0.1` | Initial learning rate |
| `--momentum` | `0.9` | SGD momentum |
| `--weight-decay` | `1e-4` | Weight decay |
| `--lr-adjust` | `step` | LR schedule: `linear` / `step` |
| `--step-ratio` | `0.01` | Step-decay ratio (LR multiplied by this every 30 epochs) |
| `--beta` | `0.6` | Coefficient of the computation-cost regularization term |
| `--computation_cost` | `True` | Whether to use computation cost as a regularization term |
| `--minimum` | `100` | Minimum threshold used for early-exit / skipping decision |
| `--loss` | `xent` | Loss function: `xent` / `adjust` |
| `--resume` | (see file) | Path to a checkpoint to resume from |
| `--save-folder` | `save_checkpoints` | Directory for checkpoints |
| `-j, --workers` | `4` | Number of data-loading workers |
| `-p, --print-freq` | `1` | Logging frequency (in batches) |
| `--pretrained` | `False` | Whether to load ImageNet-pretrained weights |

## Usage

```
# Install dependencies
pip install torch torchvision numpy

# Train the DenseNet + ACS-GN model on CIFAR-10
python train_base_channel_skip_new_gate.py train densenet40 --dataset cifar10 --epochs 120 --lr 0.1 -b 128

# Train on CIFAR-100
python train_base_channel_skip_new_gate.py train densenet40 --dataset cifar100 --epochs 120 --lr 0.1 -b 128

# Evaluate a trained checkpoint
python train_base_channel_skip_new_gate.py test densenet40 --dataset cifar10 --resume /path/to/model_best.pth.tar
```

`g_resnext.py` is a standalone model file and is not imported by the training script. To use it, import the factory functions directly in your own training code:

```python
from g_resnext import g_resnext50, g_resnext101

model = g_resnext50(num_classes=10)
logits, U_regularizer = model(x)
```

Notes:
- The default dataset paths in `train_base_channel_skip_new_gate.py` are hardcoded to Linux paths (e.g. `/home/zmx/skipnet-master/data`). Please modify them to point to your local data directory before running.
- The training script uses `torch.nn.DataParallel(model).cuda()`, so a CUDA-capable GPU is required.

## Citation

If you find ACS useful or relevant to your project and research, please kindly cite our paper:

```
@article{zou2023dynamic,
  title={Dynamic deep neural network inference via adaptive channel skipping},
  author={Zou, Meixia and Li, Xiuwen and Fang, Jinzheng and Wen, Hong and Fang, Weiwei},
  journal={Turkish Journal of Electrical Engineering and Computer Sciences},
  volume={31},
  number={5},
  pages={828--843},
  year={2023}
}
```

## Acknowledgement

Special thanks to the authors of [DDI](https://arxiv.org/abs/1907.04523) and [DGConv](https://arxiv.org/abs/1908.05867) for their kindly help.

## Contact

Meixia Zou (19120460@bjtu.edu.cn)

> Please note that the open source code in this repository was mainly completed by the graduate student author during his master's degree study. Since the author did not continue to engage in scientific research work after graduation, it is difficult to continue to maintain and update these codes. We sincerely apologize that these codes are for reference only.
