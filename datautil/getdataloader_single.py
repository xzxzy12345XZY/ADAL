# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# coding=utf-8
import numpy as np
from torch.utils.data import DataLoader

import datautil.actdata.util as actutil
from datautil.util import combindataset, subdataset

import datautil.actdata.cross_people as cross_people

task_act = {'cross_people': cross_people}


def get_dataloader(args, tr, val, tar):  # 函数定义 args：包含配置参数的对象，包括批次大小、工作线程数等。 tr：训练数据集（train）。 val：验证数据集（validation）。 tar：目标数据集（测试集，target）。
    # train_loader = DataLoader(dataset=tr, batch_size=args.batch_size,  # 创建训练数据加载器（打乱顺序） dataset=tr：指定数据加载器使用的训练数据集。
    #                           num_workers=args.N_WORKERS, drop_last=False, shuffle=True)  #  batch_size=args.batch_size：每个批次的样本数量，由 args.batch_size 指定。  num_workers=args.N_WORKERS：加载数据时使用的工作线程数，由 args.N_WORKERS 指定，可以加快数据加载的速度。 drop_last=False：当数据集不能整除批次大小时，是否丢弃最后一个不完整的批次。设置为 False，表示保留这个批次。 shuffle=True：是否在每个 epoch 开始时打乱数据顺序。设置为 True，以确保每次训练时数据顺序是随机的，从而提高模型的泛化能力。
    train_loader = DataLoader(dataset=tr, batch_size=args.batch_size, drop_last=False, shuffle=True)
    # train_loader_noshuffle = DataLoader(   # shuffle=False：与之前的加载器不同，这个加载器不会打乱数据顺序。数据按照原始顺序加载。
    #     dataset=tr, batch_size=args.batch_size, num_workers=args.N_WORKERS, drop_last=False, shuffle=False)
    train_loader_noshuffle = DataLoader(  # shuffle=False：与之前的加载器不同，这个加载器不会打乱数据顺序。数据按照原始顺序加载。
        dataset=tr, batch_size=args.batch_size, drop_last=False, shuffle=False)
    # valid_loader = DataLoader(dataset=val, batch_size=args.batch_size,  # 作用：创建一个验证数据加载器 valid_loader。
    #                           num_workers=args.N_WORKERS, drop_last=False, shuffle=False)
    valid_loader = DataLoader(dataset=val, batch_size=args.batch_size, drop_last=False, shuffle=False)
    # target_loader = DataLoader(dataset=tar, batch_size=args.batch_size,  # 作用：创建一个目标数据加载器 target_loader，通常用于测试集。
    #                            num_workers=args.N_WORKERS, drop_last=False, shuffle=False)
    target_loader = DataLoader(dataset=tar, batch_size=args.batch_size, drop_last=False, shuffle=False)
    # return train_loader, train_loader_noshuffle, valid_loader, target_loader
    return train_loader, train_loader_noshuffle, valid_loader, target_loader


def get_act_dataloader(args):  # get_act_dataloader(args) 函数的目标是根据给定的任务参数 (args)，生成训练集、验证集和目标集的数据加载器（dataloader），用于在跨域（如跨参与者）任务中训练和评估模型。此设置有助于评估模型在处理分布变化（如 OOD 问题）时的泛化能力，这是论文中主要研究的问题。
    source_datasetlist = []  # 作用：初始化两个空列表 source_datasetlist 和 target_datalist，用于分别存储源域数据集（用于训练）和目标域数据集（用于测试）。
    target_datalist = []  # 原因：区分源域和目标域是为了在训练和测试时分别使用不同的数据。源域数据集用于模型训练，而目标域数据集用于模型泛化性能的测试。
    pcross_act = task_act[args.task]  # 作用：根据任务名称（如 cross_people）从 task_act 字典中获取相应的模块。
    # 原因：动态选择合适的数据处理模块，可以实现不同任务的代码重用和灵活性。
    tmpp = args.act_people[args.dataset]  # 作用：从参数中获取当前数据集的参与者分组信息，并设置域的数量为参与者组的数量。  # 原因：每个组代表一个域（domain），例如不同的参与者群体。了解域的数量有助于后续的数据加载和模型训练时的设置。
    args.domain_num = len(tmpp)  # 域的数量
    for i, item in enumerate(tmpp):  # 作用：遍历每个参与者组，为每个组创建一个 ActList 对象，这些对象代表数据集。  # 原因：ActList 类根据提供的参与者组加载相应的数据，应用必要的数据转换（transform），并准备好数据用于训练或测试。
        tdata = pcross_act.ActList(
            args, args.dataset, args.data_dir, item, i, transform=actutil.act_train()) # 传入args、args.dataset='emg'、args.data_dir='./data/'
        if i in args.test_envs:
            target_datalist.append(tdata)
        else:
            source_datasetlist.append(tdata)  # 将 tdata 添加到 source_datasetlist 中。 # 原因：将当前数据集标记为源域（训练集），用于训练模型。这些数据来自于非测试环境的参与者组，是模型在训练时看到的数据。
            if len(tdata)/args.batch_size < args.steps_per_epoch:  # 作用：检查当前数据集的大小是否足以支持当前的每个 epoch 步数（steps_per_epoch）。
                args.steps_per_epoch = len(tdata)/args.batch_size
    rate = 0.2  # # 作用：设置验证集的比例为 20%。 # 原因：训练集和验证集的划分是深度学习模型训练中的常见步骤，验证集用于评估模型在训练过程中的性能，以帮助防止过拟合。
    args.steps_per_epoch = int(args.steps_per_epoch*(1-rate))  # 作用：调整 steps_per_epoch 的值，将其减少为原来的 80%。 # 原因：因为从源域数据集中划分出了一部分作为验证集（20%），实际用于训练的数据量减少了。因此，训练过程中每个 epoch 的步数也应相应减少，以反映训练数据量的变化。
    tdata = combindataset(args, source_datasetlist)  # 作用：将所有源域的数据集组合成一个统一的数据集对象 tdata。 # 原因：多个源域数据集需要合并在一起，形成一个整体数据集，用于训练。combindataset 函数将这些数据集整合，方便后续的训练集和验证集划分。
    l = len(tdata.labels)  # 作用：获取合并后的源域数据集 tdata 中的样本数量。 # 原因：需要知道数据集的总样本数量，以便在后续步骤中按照比例划分训练集和验证集。
    indexall = np.arange(l)  # 作用：创建一个包含所有样本索引的 numpy 数组 indexall。 # 原因：在数据集划分过程中，使用索引列表来随机分割数据，以确保训练集和验证集的随机性。
    np.random.seed(args.seed)  # np.random.seed(args.seed)：设置随机种子，以确保随机操作（如打乱数据）的可重复性。
    # np.random.shuffle(indexall)  # np.random.shuffle(indexall)：随机打乱索引数组 indexall。  # 原因：设置随机种子后，每次运行代码时，数据划分的随机性是一致的。打乱索引则是为了在后续划分训练集和验证集时确保样本是随机选择的。
    # ted = int(l*rate)  # 作用： ted = int(l * rate)：计算验证集的样本数量。
    # ==================0831------------------------------
    # indextr, indexval = indexall[ted:], indexall[:ted]  # indextr, indexval = indexall[ted:], indexall[:ted]：将索引数组 indexall 按照 ted 划分为训练集索引 indextr 和验证集索引 indexval。 原因：按设定的比例划分数据集，以确保训练集和验证集的比例与 rate 参数一致。
    indextr, indexval = indexall[:], indexall[:]  # indextr, indexval = indexall[ted:], indexall[:ted]：将索引数组 indexall 按照 ted 划分为训练集索引 indextr 和验证集索引 indexval。 原因：按设定的比例划分数据集，以确保训练集和验证集的比例与 rate 参数一致。
    # ==================0831------------------------------
    tr = subdataset(args, tdata, indextr)  # tr = subdataset(args, tdata, indextr)：根据训练集索引 indextr 从 tdata 中提取训练集 tr。
    val = subdataset(args, tdata, indexval)  # val = subdataset(args, tdata, indexval)：根据验证集索引 indexval 从 tdata 中提取验证集 val。 # 原因：使用索引提取子数据集，分别生成训练集和验证集。
    targetdata = combindataset(args, target_datalist)  # 作用：将所有目标域的数据集（测试集）组合成一个统一的目标数据集 targetdata。# 原因：目标域数据集也可能来自多个组，需要合并成一个整体，便于模型在训练结束后进行统一的测试和评估。
    train_loader, train_loader_noshuffle, valid_loader, target_loader = get_dataloader(  # 作用：调用 get_dataloader 函数，将训练集、验证集和目标数据集转换为数据加载器（dataloader）。
        args, tr, val, targetdata) # 参数说明： args：传递配置参数； tr：训练集； val：验证集；targetdata：目标数据集（测试集）。原因：将数据集封装成数据加载器，使得在训练和评估过程中能够高效地批量读取数据。
    return train_loader, train_loader_noshuffle, valid_loader, target_loader, tr, val, targetdata # 作用：返回训练集加载器、验证集加载器和目标集加载器，以及对应的训练集、验证集和目标数据集对象。 原因：提供数据加载器和数据集对象用于后续的模型训练和评估。数据加载器封装了数据的读取和预处理逻辑，便于在训练循环中使用。
