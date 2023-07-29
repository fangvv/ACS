### 自己的门控结构 CBAM

### 1×1/2  3×3/2  3×3/1 加SE

import sys
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import math
import os
import random

BatchNorm = nn.BatchNorm2d


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']


def aggregate(gate, D, I, K, sort=False):
    if sort:
        _, ind = gate.sort(descending=True)

        gate = gate[:, ind[0, :]]  # [2,5]


    U = [(gate[0, i] * D + gate[1, i] * I) for i in range(K)]


    while len(U) != 1:
        temp = []
        for i in range(0, len(U) - 1, 2):
            temp.append(kronecker_product(U[i], U[i + 1]))
        if len(U) % 2 != 0:
            temp.append(U[-1])
        del U
        U = temp

    return U[0], gate


def kronecker_product(mat1, mat2):
    return torch.ger(mat1.view(-1), mat2.view(-1)).reshape(*(mat1.size() + mat2.size())).permute(
        [0, 2, 1, 3]).reshape(mat1.size(0) * mat2.size(0), mat1.size(1) * mat2.size(1))


class DGConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, sort=False):
        super(DGConv2d, self).__init__()
        self.register_buffer('D', torch.eye(2))

        self.register_buffer('I', torch.ones(2, 2))
        self.K = int(math.log2(in_channels))

        eps = 1e-8

        gate_init = [eps * random.choice([-1, 1]) for _ in range(self.K)]


        self.register_parameter('gate', nn.Parameter(torch.Tensor(gate_init)))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=bias)


        self.in_channels = in_channels

        self.out_channels = out_channels

        self.sort = sort  ##false



    def forward(self, x):
        setattr(self.gate, 'org', self.gate.data.clone())

        self.gate.data = (self.gate.data > 0).float().detach() - \
            self.gate.data.detach() + self.gate.data


        U_regularizer = 2 ** (self.K + torch.sum(self.gate))


        gate = torch.stack((1 - self.gate, self.gate))


        self.gate.data = self.gate.org


        U, gate = aggregate(gate, self.D, self.I, self.K, sort=self.sort)

        masked_weight = self.conv.weight * U.view(self.out_channels, self.in_channels, 1, 1)


        x = F.conv2d(x, masked_weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation)
        return x, U_regularizer


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return DGConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=1, bias=False)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


#

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


def densenet40(pretrained=True, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 6, 6),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet201'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.num_input_features = num_input_features
        self.dense_module = nn.Sequential(OrderedDict([('norm1', nn.BatchNorm2d(num_input_features)),
                                                       ('relu1', nn.ReLU(inplace=True)),
                                                       ('conv1', nn.Conv2d(num_input_features, bn_size *
                                                                           growth_rate, kernel_size=1, stride=1,
                                                                           bias=False)),
                                                       ('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
                                                       ('relu2', nn.ReLU(inplace=True)),
                                                       ('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                                                           kernel_size=3, stride=1, padding=1,
                                                                           bias=False))]))

        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = self.dense_module(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.trans_module = nn.Sequential(OrderedDict([('norm', nn.BatchNorm2d(num_input_features)),
                                                       ('relu', nn.ReLU(inplace=True)),
                                                       ('conv', nn.Conv2d(num_input_features, num_output_features,
                                                                          kernel_size=1, stride=1, bias=False)),
                                                       ('pool', nn.AvgPool2d(kernel_size=2, stride=2))]))

    def forward(self, x):
        return self.trans_module(x)


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=2, drop_rate=0, num_classes=10, block=BasicBlock):

        super(DenseNet, self).__init__()

        # First convolution
        self.growth_rate = growth_rate
        self.base_layer = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        self.block_config = block_config
        num_features = num_init_features

        # denseblock 0
        for i in range(block_config[0]):
            setattr(self, 'denseblock0_%s' % i,
                    self._make_layer(i, i + 1, num_features, growth_rate, bn_size, drop_rate))

            gate = NewGate(pool_size=56, channel=num_features + (i + 1) * growth_rate, out_channels=growth_rate)

            setattr(self, 'denseblock0_%s_gate' % i, gate)

        num_features = (num_features + block_config[0] * growth_rate)  # 256

        self.trans0 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2  # 128

        # denseblock 1
        for i in range(block_config[1]):
            setattr(self, 'denseblock1_%s' % i,
                    self._make_layer(i, i + 1, num_features, growth_rate, bn_size, drop_rate))

            gate = NewGate(pool_size=28, channel=num_features + (i + 1) * growth_rate, out_channels=growth_rate)

            setattr(self, 'denseblock1_%s_gate' % i, gate)

        num_features = (num_features + block_config[1] * growth_rate)  # 512

        self.trans1 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2  # 256

        # denseblock 2
        for i in range(block_config[2]):
            setattr(self, 'denseblock2_%s' % i,
                    self._make_layer(i, i + 1, num_features, growth_rate, bn_size, drop_rate))

            gate = NewGate(pool_size=14, channel=num_features + (i + 1) * growth_rate, out_channels=growth_rate)

            setattr(self, 'denseblock2_%s_gate' % i, gate)

        num_features = (num_features + block_config[2] * growth_rate)
        self.trans2 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        # Final batch norm
        self.bn_norm = nn.BatchNorm2d(num_features)

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    #             elif isinstance(m, nn.Linear):
    #                 nn.init.constant_(m.bias, 0)

    def _make_layer(self, front_layer_idx, back_layer_index, num_input_features, growth_rate, bn_size, drop_rate):
        modules = []
        for i in range(front_layer_idx, back_layer_index):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            modules.extend([layer])
        return nn.Sequential(*modules)

    def forward(self, x):
        features = self.base_layer(x)

        masks = []
        gprobs = []

        new_features = getattr(self, 'denseblock0_0')(features)

        features = torch.cat([features, new_features], 1)

        mask, gprob = getattr(self, 'denseblock0_0_gate')(features)


        gprobs.append(gprob)
        masks.append(mask.squeeze())

        # loop for denseblock 0
        for i in range(1, self.block_config[0]):
            #             print(mask.size())

            new_features = getattr(self, 'denseblock0_{}'.format(i))(features)

            new_features = mask.expand_as(new_features) * new_features

            features = torch.cat([features, new_features], 1)

            mask, gprob = getattr(self, 'denseblock0_{}_gate'.format(i))(features)



            gprobs.append(gprob)
            masks.append(mask.squeeze())

        features = self.trans0(features)

        # loop for denseblock 1
        for i in range(self.block_config[1]):
            # print(mask.size())

            new_features = getattr(self, 'denseblock1_{}'.format(i))(features)

            new_features = mask.expand_as(new_features) * new_features

            features = torch.cat([features, new_features], 1)

            mask, gprob = getattr(self, 'denseblock1_{}_gate'.format(i))(features)

            gprobs.append(gprob)
            masks.append(mask.squeeze())

        features = self.trans1(features)

        # loop for denseblock 2
        for i in range(self.block_config[2]):
            new_features = getattr(self, 'denseblock2_{}'.format(i))(features)

            new_features = mask.expand_as(new_features) * new_features

            features = torch.cat([features, new_features], 1)

            mask, gprob = getattr(self, 'denseblock2_{}_gate'.format(i))(features)

            gprobs.append(gprob)
            masks.append(mask.squeeze())

        features = self.trans2(features)

        features = self.bn_norm(features)

        out = F.relu(features, inplace=True)

        out = F.avg_pool2d(out, kernel_size=7).view(features.size(0), -1)
        out = self.classifier(out)
        return out, masks, gprobs


########################################
# DenseNet with New Gate     #
########################################

class NewGate(nn.Module):
    """ one 1x1 conv followed by a 3x3 conv (stride=1) layer """

    def __init__(self, pool_size=5, channel=10, out_channels=32):
        super(NewGate, self).__init__()
        self.pool_size = pool_size
        self.channel = channel
        self.activate = False

        self.conv1 = nn.Conv2d(channel, 32, kernel_size=1, stride=2, bias=False)
        self.conv2 = conv3x3(32, 32, stride=2)
        self.se = SELayer(32, reduction=16)
        self.bn1 = nn.BatchNorm2d(32)

        self.relu1 = nn.ReLU(inplace=True)


        self.conv3 = conv3x3(32, out_channels, stride=1)

        pool_size = math.floor(pool_size / 2 + 0.5)

        self.avg_layer = nn.AvgPool2d(pool_size)

        self.linear_layer = nn.Conv2d(in_channels=out_channels, out_channels=32,
                                      kernel_size=1, stride=1)

        self.prob_layer = nn.Sigmoid()
        self.logprob = nn.LogSoftmax(dim=1)

    def forward(self, x):
        U_regularizer_sum = 0
        if isinstance(x, tuple):
            x, U_regularizer_sum = x[0], x[1]

        x = self.conv1(x)
        x, U_regularizer = self.conv2(x)

        # x = self.se(x)
        x = self.bn1(x)

        x = self.relu1(x)

        x, U_regularizer = self.conv3(x)

        x = self.avg_layer(x)

        x = self.linear_layer(x)

        x = x.view(x.size(0), -1)

        prob = self.prob_layer(x)


        logprob = self.logprob(x)

        # discretizes
        x = (prob > 0.5).float().detach() - \
            prob.detach() + prob


        x = x.view(x.size(0), -1, 1, 1)

        return x, logprob


if __name__ == '__main__':
    net = NewGate()
    #     net = FeedforwardGateI()
    net = densenet41()
    print(net)
