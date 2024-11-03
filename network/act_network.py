# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch.nn as nn

var_size = {
    'emg': {
        'in_size': 8,
        'ker_size': 9,
        'fc_size': 32*44
    },
    'pads': {
        'in_size': 6,
        'ker_size': 9,
        'fc_size': 32*19
    }
}


class ActNetwork(nn.Module):  # ActNetwork 类是一个卷积神经网络（CNN），用于特定任务的特征提取。它通过一系列卷积层、批归一化、激活函数和池化层来处理输入数据，并最终将输出特征展平成一个向量。这种网络结构通常用于处理图像或多维传感器数据。以下是对 ActNetwork 类及其代码的详细解释：
    def __init__(self, taskname):  # 初始化方法：定义 __init__ 方法，用于创建 ActNetwork 类的实例。
        super(ActNetwork, self).__init__()
        self.taskname = taskname  # 作用：将任务名称 taskname 保存为类的属性。  原因：任务名称用于在类中引用，选择与特定任务相关的网络配置和超参数。
        self.conv1 = nn.Sequential(  # 定义第一层卷积模块 定义 conv1：创建一个卷积模块 conv1，包含卷积层、批归一化层、ReLU 激活函数和最大池化层。
            nn.Conv2d(in_channels=var_size[taskname]['in_size'], out_channels=16, kernel_size=(  # nn.Conv2d
                1, var_size[taskname]['ker_size'])),  # in_channels=var_size[taskname]['in_size']：输入通道数，来自 var_size 字典，根据 taskname 提供的任务特定配置。
            nn.BatchNorm2d(16),  # out_channels=16：输出通道数，固定为 16。
            nn.ReLU(),  # kernel_size=(1, var_size[taskname]['ker_size'])：卷积核大小，高度为 1，宽度根据任务配置决定。
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)  # nn.BatchNorm2d(16)：对卷积层输出进行批归一化，标准化每个批次的数据，减少训练中
        )  # nn.ReLU()：应用 ReLU 激活函数，引入非线性特征。 nn.MaxPool2d(kernel_size=(1, 2), stride=2)：最大池化层，核大小为 (1, 2)，将特征进行下采样，减少特征图的尺寸。 原因：通过这组层的组合，conv1 模块能够提取输入数据的局部特征，进行归一化和降采样，为后续的卷积层提供更有意义的特征。
        self.conv2 = nn.Sequential(  # 定义 conv2：创建第二个卷积模块 conv2，与 conv1 结构类似，但通道数增加到 32。
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(  # in_channels=16：输入通道数为 16，与 conv1 的输出通道数匹配。
                1, var_size[taskname]['ker_size'])),  # out_channels=32：输出通道数为 32。
            nn.BatchNorm2d(32),  # kernel_size=(1, var_size[taskname]['ker_size'])：卷积核大小与 conv1 一致，按任务配置选择。
            nn.ReLU(),  # nn.BatchNorm2d(32)：对卷积层输出进行批归一化。 nn.ReLU()：应用 ReLU 激活函数。
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)  # nn.MaxPool2d(kernel_size=(1, 2), stride=2)：最大池化层，下采样特征图。
        )  # 原因：conv2 模块在 conv1 提取的基础特征上进一步提取更复杂的特征，同时通过增加通道数，增强了特征表达能力。
        self.in_features = var_size[taskname]['fc_size']  # 设置全连接层输入特征大小 作用：设置全连接层输入特征大小，作为网络的最终特征维度。  原因：根据任务配置的特征维度，为后续的全连接层或其他处理步骤提供正确的输入大小。

    def forward(self, x):  # 定义 forward 方法：实现网络的前向传播。
        x = self.conv2(self.conv1(x))  # x：输入数据张量，通常形状为 [batch_size, channels, height, width]。 x = self.conv2(self.conv1(x))：将输入数据依次传递通过 conv1 和 conv2 卷积模块。#  self.conv1(x)：通过第一层卷积模块提取初步特征。 # self.conv2(...)：通过第二层卷积模块进一步提取特征。
        x = x.view(-1, self.in_features)  # x = x.view(-1, self.in_features)：将卷积后的特征图展平成形状为 [batch_size, self.in_features] 的二维张量。 # view 操作是 PyTorch 中用于改变张量形状的函数，在这里它将多维特征图展平为一维向量，适配全连接层输入要求。
        return x  # 原因：前向传播过程中，通过卷积和展平操作，将输入数据转换为固定大小的特征向量，为后续的任务（如分类、回归）提供输入特征。

# 总结
# ActNetwork 类实现了一个简单的卷积神经网络结构，用于特定任务的特征提取。通过多层卷积、批归一化、激活和池化操作，网络能够有效地处理和提取输入数据中的重要特征。
# 通过任务名称 taskname，类可以灵活地配置输入通道、卷积核大小和最终特征维度，适应不同任务的需求。
# forward 方法定义了数据的前向流动路径，从输入到特征提取再到展平为特征向量，为后续的进一步处理提供了接口。
#