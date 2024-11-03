# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from datautil.actdata.util import *
from datautil.util import mydataset, Nmax
import numpy as np
import torch


class ActList(mydataset):
    def __init__(self, args, dataset, root_dir, people_group, group_num, transform=None, target_transform=None, pclabels=None, pdlabels=None, shuffle_grid=True):
        super(ActList, self).__init__(args)
        self.domain_num = 0  # self.domain_num：初始化域编号为 0。
        self.dataset = dataset  # self.dataset：设置数据集名称。
        self.task = 'cross_people'  # self.task：任务类型设置为 'cross_people'，表示跨参与者的任务。
        self.transform = transform  # self.transform 和 self.target_transform：数据和标签的转换方法。
        self.target_transform = target_transform # x：输入数据（如传感器数据）。 cy：类别标签。  py：参与者标签。 sy：传感器位置标签。 # 原因：在跨参与者任务中，需要根据参与者和传感器位置加载相应的数据。
        x, cy, py, sy = loaddata_from_numpy(self.dataset, self.task, root_dir, args)  # 加载数据：调用 loaddata_from_numpy 函数，从指定的根目录加载数据。
        self.people_group = people_group  # 设置参与者组：保存参与者组信息，用于后续数据筛选。
        self.position = np.sort(np.unique(sy))  # 设置传感器位置：提取并排序传感器位置标签中的唯一值，用于确定所有数据的位置布局。
        self.comb_position(x, cy, py, sy)  # 调用 comb_position：根据参与者和传感器位置组合数据，生成符合模型输入要求的数据矩阵。
        self.x = self.x[:, :, np.newaxis, :]  # 增加维度：在第三个维度增加一个新轴，使数据符合特定的输入格式（如 [batch, channels, height, width] 格式中的维度匹配）。
        self.transform = None  # 转换为张量：将数据转换为 torch 张量，并转换为浮点数，以便与 PyTorch 的模型兼容。
        self.x = torch.tensor(self.x).float()  #
        if pclabels is not None:  # 设置标签  # 若提供了 pclabels，则直接使用；否则，初始化为 -1。 # 原因：参与者标签可能用于特定的损失函数或分析，但如果没有特定标签，则使用默认值表示未定义。
            self.pclabels = pclabels
        else:
            self.pclabels = np.ones(self.labels.shape)*(-1)
        if pdlabels is not None:  # 设置域标签 (pdlabels)： # 若提供了 pdlabels，则直接使用；否则，初始化为 0。  #
            self.pdlabels = pdlabels  # 原因：域标签用于表示数据来源的不同域（如不同参与者群体），若没有指定则默认为单一域。
        else:
            self.pdlabels = np.ones(self.labels.shape)*(0)
        self.tdlabels = np.ones(self.labels.shape)*group_num  # 设置任务域标签 (tdlabels)：所有数据的任务域标签均设置为当前组编号 group_num。
        self.dlabels = np.ones(self.labels.shape) * \
            (group_num-Nmax(args, group_num))
        # 设置动态域标签 (dlabels)：根据 group_num 和 Nmax 函数计算动态域标签。 # Nmax(args, group_num)：计算域调整值，用于动态调整标签。 # 原因：这些标签可以用于对抗性训练或其他需要区分不同域的数据处理中，增强模型的泛化能力。
    def comb_position(self, x, cy, py, sy):  # comb_position 方法 # 遍历参与者组：根据 people_group 中的每个参与者，筛选出属于该参与者的数据。
        for i, peo in enumerate(self.people_group):  # 筛选数据：
            index = np.where(py == peo)[0]  # np.where(py == peo)[0]：找到当前参与者的所有数据索引。
            # 判断 index 是否为空
            if index.size == 0:  # 使用 .size 检查数组的元素数量
                raise ValueError(f"No data found for participant: {peo}")
            tx, tcy, tsy = x[index], cy[index], sy[index]  # tx, tcy, tsy：分别获取当前参与者的输入数据、类别标签和位置标签。
            for j, sen in enumerate(self.position):  # 遍历传感器位置： # 作用：根据位置标签，将属于相同位置的数据按顺序拼接。
                index = np.where(tsy == sen)[0]  # 数据拼接： np.where(tsy == sen)[0]：获取当前传感器位置的所有数据索引。
                if j == 0:
                    ttx, ttcy = tx[index], tcy[index]  # ttx 和 ttcy：分别拼接当前参与者的输入数据和类别标签。
                else:
                    ttx = np.hstack((ttx, tx[index]))
            if i == 0:  # 整合所有参与者数据： 如果是第一个参与者，直接赋值。 否则，将当前参与者的数据和标签与之前的数据进行垂直（np.vstack）和水平拼接（np.hstack）
                self.x, self.labels = ttx, ttcy  # 原因：通过这种方式，逐步整合所有参与者的数据和标签，形成完整的数据集。
            else:
                self.x, self.labels = np.vstack(
                    (self.x, ttx)), np.hstack((self.labels, ttcy))

    def set_x(self, x):  # set_x 方法  # 作用：提供一个方法用于外部修改或重新设置 self.x 数据。
        self.x = x  # 原因：在某些情况下，可能需要对 self.x 进行后续处理或替换，因此提供一个简便的方法直接修改数据。

    # 结合论文逻辑的总结
    # 跨域学习：ActList 类和其中的逻辑支持跨参与者的数据处理，符合论文中关于跨域（跨参与者）学习和泛化的研究目标。
    # 数据预处理与标签设置：通过处理和组合参与者和传感器位置的数据，以及灵活设置标签，这些步骤为后续训练模型提供了一个结构化且灵活的数据输入流程。
    # 对抗性训练准备：标签设置和数据组合方式支持对抗性训练或域不变特征学习，这些是论文中增强模型泛化能力的关键策略。
    # 这些代码为训练任务构建了一个灵活的数据加载和处理框架，有助于在跨域学习中实现对复杂分布变化的应对和泛化能力的提升。
