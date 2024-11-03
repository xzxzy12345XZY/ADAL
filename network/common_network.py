# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch.nn as nn
import torch.nn.utils.weight_norm as weightNorm


class feat_bottleneck(nn.Module):  # 定义类：feat_bottleneck 是一个瓶颈层模块，继承自 nn.Module。 目的：用于在特征提取阶段减少特征维度，或对特征进行变换，以便后续的分类任务。
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):  # 初始化方法：定义 __init__ 方法，用于创建 feat_bottleneck 类的实例。feature_dim：输入特征的维度。 bottleneck_dim=256：瓶颈层输出特征的维度，默认为 256。 type="ori"：瓶颈层的类型，决定是否在输出中应用批归一化。
        super(feat_bottleneck, self).__init__()  # 调用父类初始化：通过 super() 调用 nn.Module 的初始化方法，确保继承的属性和方法被正确初始化。
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)  # 作用：定义一个批归一化层 bn。 bottleneck_dim：批归一化的输入维度，与瓶颈层输出一致。 affine=True：启用缩放和平移参数，使批归一化更灵活。 原因：批归一化可以稳定训练过程，减少梯度消失或爆炸的风险。
        self.relu = nn.ReLU(inplace=True)  # 作用：定义一个 ReLU 激活函数 relu，用于引入非线性。 inplace=True：直接在原地修改输入，节省内存。 原因：激活函数引入非线性，有助于模型学习复杂的模式和关系。
        self.dropout = nn.Dropout(p=0.5)  # 作用：定义一个 Dropout 层 dropout，用于防止过拟合。 参数： p=0.5：在训练过程中以 50% 的概率随机丢弃一些神经元的输出。 原因：Dropout 通过抑制神经元协同适应，减少过拟合，提高模型的泛化能力。
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)  # 作用：定义一个线性层 bottleneck，用于将输入特征从 feature_dim 映射到 bottleneck_dim。 # 参数： feature_dim：输入特征的维度。  bottleneck_dim：瓶颈层输出的维度。 原因：线性层用于特征维度的变换或压缩，为后续的分类或其他任务提供合适的特征表示。
        self.type = type  # 保存类型参数：将 type 参数保存为类属性，以便在前向传播中使用。  原因：控制是否应用批归一化。

    def forward(self, x):  # 定义 forward 方法：实现网络的前向传播。  参数： x：输入数据张量，通常形状为 [batch_size, feature_dim]。
        x = self.bottleneck(x)  # 操作： self.bottleneck(x)：通过瓶颈层变换输入特征。 原因：将输入特征映射到瓶颈空间，为后续处理做好准备。
        if self.type == "bn":  # 条件操作：如果 type 参数为 "bn"，则在输出上应用批归一化。 原因：批归一化可以标准化特征，使其在训练过程中更加稳定和高效。
            x = self.bn(x)
        return x  # 返回结果：返回变换后的特征张量。


class feat_classifier(nn.Module):  # feat_classifier 类  定义类：feat_classifier 是一个分类器模块，继承自 nn.Module。 目的：用于对提取到的特征进行分类。
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):  # 初始化方法：定义 __init__ 方法，用于创建 feat_classifier 类的实例。 class_num：分类器的输出类别数。 bottleneck_dim=256：输入特征的维度，默认为 256。 type="linear"：分类器的类型，默认为线性分类器。
        super(feat_classifier, self).__init__()  # 调用父类初始化：通过 super() 调用 nn.Module 的初始化方法。
        self.type = type  # 定义分类器组件  保存类型参数：将 type 参数保存为类属性，以便在前向传播中使用。  原因：控制分类器的类型。
        if type == 'wn':  # 条件定义分类器： if type == 'wn':：如果类型是 'wn'，则使用带权重归一化的线性层。
            self.fc = weightNorm(  # nn.Linear(bottleneck_dim, class_num)：定义一个线性层，将输入特征映射到类别数空间。
                nn.Linear(bottleneck_dim, class_num), name="weight")  # weightNorm：一种对线性层参数进行归一化的方式，可以稳定训练并加速收敛。
        else:  # else:：否则，定义一个普通的线性分类器。  原因：带权重归一化的分类器可能在某些任务中表现更好，因此提供了这个选项。
            self.fc = nn.Linear(bottleneck_dim, class_num)  #  原因：带权重归一化的分类器可能在某些任务中表现更好，因此提供了这个选项。

    def forward(self, x):  #  定义 forward 方法：实现分类器的前向传播。
        x = self.fc(x)  # 参数：  x：输入特征张量，形状通常为 [batch_size, bottleneck_dim]。 操作：  self.fc(x)：通过全连接层将特征映射到类别空间，得到分类结果。
        return x  # 返回结果：返回分类器的输出，即每个类别的得分。


# 总结
# feat_bottleneck 类：用于特征压缩或变换，包含线性变换、批归一化、激活和 Dropout 层。通过这些组件的组合，feat_bottleneck 可以对输入特征进行有效的预处理，为后续分类任务做好准备。
# feat_classifier 类：用于分类任务，将瓶颈层输出的特征映射到分类空间。它提供了普通线性分类器和带权重归一化的分类器两种选项，以满足不同的任务需求。
# 设计模式：这些模块通过条件参数控制模块行为（如批归一化和权重归一化），增加了网络的灵活性和适应性，可以根据任务的需求进行调整。
# 545