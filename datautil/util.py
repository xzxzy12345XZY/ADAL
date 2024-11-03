# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import numpy as np
import torch

#
def Nmax(args, d):  # 定义函数 Nmax：该函数接受两个参数： args：一个包含配置和参数的对象，这里特别关注其中的 test_envs 属性。 ：一个整数值，用于比较 test_envs 列表中的元素。
    for i in range(len(args.test_envs)): # for 循环：遍历 args.test_envs 列表的索引。args.test_envs 是一个整数列表，用于存储测试环境的编号（可能是不同的域、不同的参与者组等）。
        if d < args.test_envs[i]:  # range(len(args.test_envs))：生成从 0 到 len(args.test_envs) - 1 的索引序列，用于访问 test_envs 中的每个元素。
            return i               # 找大于 d 的元素
    return len(args.test_envs)  # 条件判断：检查 d 是否小于 args.test_envs[i]。 # 如果 d 小于当前索引 i 处的元素 args.test_envs[i]，则返回当前索引 i 原因：这是为了找到第一个大于 d 的元素的位置，这样的索引可能用于进一步的数据处理或逻辑判断。
# Nmax 函数用于确定一个值 d 在一个列表 args.test_envs 中的位置。具体来说，它在列表中寻找第一个大于 d 的值的位置索引，并返回该索引。如果 d 大于或等于列表中所有的元素，则返回列表的长度。这种操作在许多情况下可能用于域标签或索引的计算。以下是对每一行代码的详细解释：

class basedataset(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class mydataset(object):
    def __init__(self, args):
        self.x = None  # 初始化 x 为 None，用于存储数据集的输入数据（如传感器数据、图像等）。
        self.labels = None  # 初始化 labels 为 None，用于存储类别标签，通常是用于分类任务的主要目标标签。
        self.dlabels = None  # 初始化 dlabels 为 None，用于存储域标签，表示数据所属的域（如不同参与者、不同实验条件）。
        self.pclabels = None  # 初始化 pclabels 和 pdlabels 为 None，用于存储参与者标签和另一个域相关的标签，可能用于更细粒度的标签控制。
        self.pdlabels = None  # 初始化 pclabels 和 pdlabels 为 None，用于存储参与者标签和另一个域相关的标签，可能用于更细粒度的标签控制。
        self.task = None  # task 用于指定当前任务的类型（如 'cross_people'），dataset 用于指定当前使用的数据集名称。
        self.dataset = None  # 数据集名称
        self.transform = None  # transform 用于定义输入数据的预处理方法（如数据增强），target_transform 用于定义标签的预处理方法。
        self.target_transform = None
        self.loader = None  # 初始化 loader 为 None，可以用于存储数据加载器或其他数据访问方式。
        self.args = args  # 将传入的 args 参数对象存储为类属性，以便在类的方法中访问配置信息。

    def set_labels(self, tlabels=None, label_type='domain_label'):
        assert len(tlabels) == len(self.x)
        if label_type == 'pclabel':
            self.pclabels = tlabels
        elif label_type == 'pdlabel':
            self.pdlabels = tlabels
        elif label_type == 'domain_label':
            self.dlabels = tlabels
        elif label_type == 'class_label':
            self.labels = tlabels

    def set_labels_by_index(self, tlabels=None, tindex=None, label_type='domain_label'):
        if label_type == 'pclabel':
            self.pclabels[tindex] = tlabels
        elif label_type == 'pdlabel':
            self.pdlabels[tindex] = tlabels
        elif label_type == 'domain_label':
            self.dlabels[tindex] = tlabels
        elif label_type == 'class_label':
            self.labels[tindex] = tlabels

    def target_trans(self, y):
        if self.target_transform is not None:
            return self.target_transform(y)
        else:
            return y

    def input_trans(self, x):
        if self.transform is not None:
            return self.transform(x)
        else:
            return x

    def __getitem__(self, index):
        x = self.input_trans(self.x[index])

        ctarget = self.target_trans(self.labels[index])
        dtarget = self.target_trans(self.dlabels[index])
        pctarget = self.target_trans(self.pclabels[index])
        pdtarget = self.target_trans(self.pdlabels[index])
        return x, ctarget, dtarget, pctarget, pdtarget, index

    def __len__(self):
        return len(self.x)


class subdataset(mydataset):  # subdataset 类是一个自定义的数据集类，用于从现有的父数据集中提取子集。这个类通过索引 (indices) 从一个较大的数据集中选择特定的样本，生成一个包含部分数据的新数据集。这样的设计通常用于创建训练集、验证集或测试集。
    def __init__(self, args, dataset, indices):  # 目的：用于从一个较大的数据集中提取一个子集，通常用于训练、验证或测试的目的。
        super(subdataset, self).__init__(args)  # 初始化方法：定义 __init__ 方法，用于创建 subdataset 类的实例。
        self.x = dataset.x[indices]  # 作用：从父数据集的输入数据 dataset.x 中提取特定索引 indices 对应的样本。 # 原因：通过这种方式，仅选择指定的样本，形成子数据集的输入特征 self.x。这通常用于分割训练集、验证集或测试集。
        self.loader = dataset.loader  # 作用：将数据加载器 loader 设置为父数据集的加载器。 # 原因：保持与父数据集相同的加载器，以便使用相同的加载和预处理逻辑，确保一致性。
        self.labels = dataset.labels[indices]  # 作用：从父数据集的类别标签 dataset.labels 中提取指定索引 indices 对应的标签。
        self.dlabels = dataset.dlabels[indices] if dataset.dlabels is not None else None  # 作用：从父数据集的域标签 dataset.dlabels 中提取指定索引 indices 对应的标签，如果域标签存在的话。
        self.pclabels = dataset.pclabels[indices] if dataset.pclabels is not None else None  #  作用：从父数据集的参与者标签 dataset.pclabels 中提取指定索引 indices 对应的标签，如果参与者标签存在的话。
        self.pdlabels = dataset.pdlabels[indices] if dataset.pdlabels is not None else None  #  作用：从父数据集的另一域相关标签 dataset.pdlabels 中提取指定索引 indices 对应的标签，如果这些标签存在的话。
        self.task = dataset.task  # self.task：继承父数据集的任务类型（如 'cross_people'），保持子数据集的任务一致性。
        self.dataset = dataset.dataset  # self.dataset：继承父数据集的名称，标识数据集来源。
        self.transform = dataset.transform  # self.transform：继承父数据集的输入变换操作，确保输入数据的预处理一致。
        self.target_transform = dataset.target_transform  # self.target_transform：继承父数据集的标签变换操作，确保标签数据的预处理一致。


class combindataset(mydataset):  # 定义类：combindataset 继承自 mydataset，表明它是一个特殊类型的数据集类，具备从 mydataset 继承的所有基础功能，同时还实现了数据集合并的逻辑。
    def __init__(self, args, datalist):  # 初始化方法：定义 __init__ 方法，用于创建 combindataset 类的实例。
        super(combindataset, self).__init__(args)  # args：包含配置和参数的对象，用于初始化父类 mydataset 及其他属性。 调用父类初始化：super(combindataset, self).__init__(args) 调用 mydataset 的初始化方法，确保继承的属性和方法被正确初始化。
        self.domain_num = len(datalist)  # 设置域数量  # 作用：设置 domain_num 为 datalist 中数据集的数量。  # 原因：domain_num 反映了数据集中不同域的数量，例如不同参与者组的数量，这在多域学习中是一个重要的属性。
        self.loader = datalist[0].loader  # 设置数据加载器  # 作用：将数据加载器 loader 设置为 datalist 中第一个数据集的加载器。 # 原因：假设所有数据集使用相同的加载器，直接使用第一个数据集的加载器即可。数据加载器用于批量读取数据。
        xlist = [item.x for item in datalist]  # 合并数据集的特征和标签  xlist：提取每个数据集的输入数据 x，形成一个列表。
        cylist = [item.labels for item in datalist]  # cylist：提取每个数据集的类别标签 labels，形成一个列表。
        dylist = [item.dlabels for item in datalist]  # dylist：提取每个数据集的域标签 dlabels，形成一个列表。
        pcylist = [item.pclabels for item in datalist]  # pcylist：提取每个数据集的参与者标签 pclabels，形成一个列表。
        pdylist = [item.pdlabels for item in datalist]  # pdylist：提取每个数据集的另一域相关标签 pdlabels，形成一个列表。
        self.dataset = datalist[0].dataset
        self.task = datalist[0].task
        self.transform = datalist[0].transform
        self.target_transform = datalist[0].target_transform
        self.x = torch.vstack(xlist)

        self.labels = np.hstack(cylist)
        self.dlabels = np.hstack(dylist)
        self.pclabels = np.hstack(pcylist) if pcylist[0] is not None else None
        self.pdlabels = np.hstack(pdylist) if pdylist[0] is not None else None
