## ACS

This is the source code for our paper: **Dynamic Deep Neural Network Inference via Adaptive Channel Skipping**. A brief introduction of this work is as follows:

> Deep Neural Networks have recently made remarkable achievements in computer vision applications. However, the high computational requirements needed to achieve accurate inference results can be a significant barrier to deploying DNNs on resource-constrained computing devices, such as those found in the Internet-of-Things. In this work, we propose a fresh approach called Adaptive Channel Skipping (ACS) that prioritizes the identification of the most suitable channels for skipping and implements an efficient skipping mechanism during inference. We begin with the development of a new Gating Network model, ACS-GN, which employs fine-grained channel-wise skipping to enable input-dependent inference and achieve a desirable balance between accuracy and resource consumption. To further enhance the efficiency of channel skipping, we propose a Dynamic Grouping convolutional computing approach, ACS-DG, which helps to reduce the computational cost of ACS-GN. The results of our experiment indicate that ACS-GN and ACS-DG exhibit superior performance compared to existing gating network designs and convolutional computing mechanisms, respectively. When they are combined, the ACS framework results in a significant reduction of computational expenses and a remarkable improvement in the accuracy of inferences.

## Required software

PyTorch

## Acknowledgement

Special thanks to the authors of [DDI](https://arxiv.org/abs/1907.04523) and [DGConv](https://arxiv.org/abs/1908.05867) for their kindly help. 

## Contact

Meixia Zou (19120460@bjtu.edu.cn)

> Please note that the open source code in this repository was mainly completed by the graduate student author during his master's degree study. Since the author did not continue to engage in scientific research work after graduation, it is difficult to continue to maintain and update these codes. We sincerely apologize that these codes are for reference only.
