## ACS

This is the source code for our paper: **Dynamic Deep Neural Network Inference via Adaptive Channel Skipping**. A brief introduction of this work is as follows:

> Deep Neural Networks have recently made remarkable achievements in computer vision applications. However, the high computational requirements needed to achieve accurate inference results can be a significant barrier to deploying DNNs on resource-constrained computing devices, such as those found in the Internet-of-Things. In this work, we propose a fresh approach called Adaptive Channel Skipping (ACS) that prioritizes the identification of the most suitable channels for skipping and implements an efficient skipping mechanism during inference. We begin with the development of a new Gating Network model, ACS-GN, which employs fine-grained channel-wise skipping to enable input-dependent inference and achieve a desirable balance between accuracy and resource consumption. To further enhance the efficiency of channel skipping, we propose a Dynamic Grouping convolutional computing approach, ACS-DG, which helps to reduce the computational cost of ACS-GN. The results of our experiment indicate that ACS-GN and ACS-DG exhibit superior performance compared to existing gating network designs and convolutional computing mechanisms, respectively. When they are combined, the ACS framework results in a significant reduction of computational expenses and a remarkable improvement in the accuracy of inferences.

> 深度神经网络近期在计算机视觉应用领域取得了显著成就。然而，要实现精确推理结果所需的高计算量，可能成为在资源受限的计算设备（如物联网设备）上部署深度神经网络的主要障碍。本研究提出了一种名为自适应通道跳跃（ACS）的创新方法，该方法优先识别最适合跳过的通道，并在推理过程中实现高效跳跃机制。我们首先开发了新型门控网络模型ACS-GN，该模型采用细粒度通道级跳跃技术，既能实现输入依赖型推理，又能在精度与资源消耗之间达成理想平衡。为进一步提升通道跳跃效率，我们提出了动态分组卷积计算方法ACS-DG，有效降低了ACS-GN的计算成本。实验结果表明：ACS-GN在门控网络设计方面、ACS-DG在卷积计算机制方面，分别优于现有方案。当两者结合时，ACS框架能显著降低计算开销，并大幅提升推理精度。



## Required software

PyTorch

## Acknowledgement

Special thanks to the authors of [DDI](https://arxiv.org/abs/1907.04523) and [DGConv](https://arxiv.org/abs/1908.05867) for their kindly help. 

## Contact

Meixia Zou (19120460@bjtu.edu.cn)

> Please note that the open source code in this repository was mainly completed by the graduate student author during his master's degree study. Since the author did not continue to engage in scientific research work after graduation, it is difficult to continue to maintain and update these codes. We sincerely apologize that these codes are for reference only.
