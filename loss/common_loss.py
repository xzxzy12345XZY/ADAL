# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.mean(torch.sum(entropy, dim=1))
    return entropy


def Entropylogits(input, redu='mean'):  # 定义函数 Entropylogits，用于计算输入张量（input）的熵。 ## input：表示模型的输出 logits，即未经归一化的分数。 ## redu：可选参数，用于指定熵的计算方式。默认为 'mean'，表示对整个批次的熵取平均值。另一种选择是 'None'，表示对每个样本计算熵。
    input_ = F.softmax(input, dim=1)  ## 使用 softmax 函数将输入 logits 转换为概率分布。  ## 在第 1 维（即每个样本的不同类别）上计算 softmax，以确保输出的概率和为 1。 ## input_ 是一个概率分布张量，其形状与 input 相同。
    bs = input_.size(0)  ## 获取 input_ 的第一个维度大小，即批次大小（batch size）。 ## bs 表示批次中的样本数量。
    epsilon = 1e-5  ## 设置一个小的常数值 epsilon，用于防止计算对数时出现 log(0) 的数值不稳定问题。 ## 这个小数值会被加到 input_ 上，以确保其内部没有零值。
    entropy = -input_ * torch.log(input_ + epsilon)  ## 计算每个类别的熵值。 ## input_ 是一个概率分布，torch.log(input_ + epsilon) 计算每个概率的对数。 ## 逐元素计算每个样本的熵值。这里的熵值是对每个类别的概率贡献计算的。
    if redu == 'mean':  ## 检查参数 redu 是否为 'mean'。 ## 如果是 'mean'，表示需要计算整个批次的平均熵。
        entropy = torch.mean(torch.sum(entropy, dim=1))  ## 对每个样本的类别熵求和（torch.sum(entropy, dim=1)），得到每个样本的总熵。  ## 对整个批次的样本熵取平均值（torch.mean(...)），得到批次的平均熵。
    elif redu == 'None':  ## 如果 redu 参数设置为 'None'，则不会对熵值取平均，而是返回每个样本的熵值。
        entropy = torch.sum(entropy, dim=1)  ## 对每个样本的类别熵求和，得到每个样本的总熵。  ## 不对熵值取平均，直接返回每个样本的总熵。
    return entropy  ## 返回计算出的熵值。
