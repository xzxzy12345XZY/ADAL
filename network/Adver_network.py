# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from torch.autograd import Function

# 定义类：ReverseLayerF 继承自 torch.autograd.Function，用于自定义一个自动求导的操作。 目的：实现一个梯度翻转层，常用于对抗性训练中，使得特征提取器能够学习到域不变的特征。
class ReverseLayerF(Function):  # ReverseLayerF 和 Discriminator 是两个神经网络组件，常用于对抗性学习和域自适应任务中。ReverseLayerF 是一个特殊的反向层，用于梯度翻转操作，而 Discriminator 则是一个判别器，用于区分不同的域。下面是对每一行代码的详细解释：
    @staticmethod
    def forward(ctx, x, alpha):  # 定义 forward 方法：用于定义该层在前向传播时的操作。 ctx：上下文对象，用于存储信息以供反向传播时使用。 x：输入张量。 alpha：一个标量，用于控制梯度翻转的强度。
        ctx.alpha = alpha  # 操作： ctx.alpha = alpha：将 alpha 保存到上下文对象中，以便在反向传播时使用。 return x.view_as(x)：返回输入张量本身，形状保持不变。  原因：前向传播时不改变输入，只传递数据。实际效果在反向传播时体现。
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):  # 定义 backward 方法：用于定义该层在反向传播时的操作。 ctx：上下文对象，包含前向传播时保存的信息。 grad_output：从上游传递下来的梯度。
        output = grad_output.neg() * ctx.alpha  # output = grad_output.neg() * ctx.alpha：对梯度取负并乘以 alpha，实现梯度翻转。
        return output, None  # output：经过处理的梯度。  # None：表示 alpha 没有梯度。  # 原因：通过翻转梯度，使得模型在训练时对抗性地调整特征提取器的参数，迫使特征提取器提取到对判别器无用的（即域不变的）特征。


class Discriminator(nn.Module):  # Discriminator 类  定义类：Discriminator 是一个判别器模型，继承自 nn.Module。  目的：用于区分不同的域，在对抗性学习中与特征提取器进行对抗。
    def __init__(self, input_dim=256, hidden_dim=256, num_domains=4):  #  初始化方法：定义 __init__ 方法，用于创建 Discriminator 类的实例。  input_dim=256：输入特征的维度。  hidden_dim=256：隐藏层的维度。  num_domains=4：输出类别数，代表域的数量。
        super(Discriminator, self).__init__()  # 调用父类初始化：通过 super() 调用 nn.Module 的初始化方法，确保继承的属性和方法被正确初始化。
        self.input_dim = input_dim  # self.input_dim = input_dim  self.hidden_dim = hidden_dim
        self.hidden_dim = hidden_dim  # 作用：将输入维度和隐藏层维度保存为类的属性。 原因：便于在其他方法中引用这些参数。# self.input_dim = input_dim  self.hidden_dim = hidden_dim
        layers = [  # 定义网络层  定义 layers 列表：包含判别器的所有层。
            nn.Linear(input_dim, hidden_dim),  # nn.Linear(input_dim, hidden_dim)：输入层，将输入特征从 input_dim 映射到 hidden_dim。
            nn.BatchNorm1d(hidden_dim),  # nn.BatchNorm1d(hidden_dim)：对隐藏层输出进行批归一化，减少训练中的不稳定性。
            nn.ReLU(),  # nn.ReLU()：ReLU 激活函数，引入非线性。
            nn.Linear(hidden_dim, hidden_dim),  # nn.Linear(hidden_dim, hidden_dim)：第二个线性层，保持维度不变。
            nn.BatchNorm1d(hidden_dim),  # nn.BatchNorm1d(hidden_dim)：批归一化。
            nn.ReLU(),  # nn.ReLU()：ReLU 激活函数。
            nn.Linear(hidden_dim, num_domains),  # 输出层，将隐藏层输出映射到 num_domains 个类别，输出为域类别的得分。
        ]  # 原因：这些层的组合构成了一个判别器网络，用于区分输入特征的域。
        self.layers = torch.nn.Sequential(*layers)  # 构建网络  作用：使用 torch.nn.Sequential 将所有层串联起来，构成完整的判别器模型。

    def forward(self, x):  # 定义 forward 方法：实现判别器的前向传播。  # x：输入特征张量，形状通常为 [batch_size, input_dim]。
        return self.layers(x)  # self.layers(x)：通过 Sequential 定义的所有层，执行完整的前向传播。  返回：返回判别器的输出，即域分类的得分。  原因：实现前向传播，用于在训练过程中将输入特征映射到域类别。


#  总结
# ReverseLayerF 类：实现了一个梯度翻转层，在对抗性学习中翻转特征提取器的梯度，使其学习域不变特征。
# Discriminator 类：构建了一个判别器模型，通过多层感知器（MLP）对输入特征进行分类，区分输入数据的域。它通过与特征提取器的对抗性训练，帮助特征提取器提取域不变特征。
# 设计模式：两者结合用于对抗性域自适应学习，使得模型在不同域上的泛化能力更强。ReverseLayerF 调整梯度方向，而 Discriminator 试图区分不同的域，从而形成一种对抗训练的机制。
#
