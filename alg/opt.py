# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

#  get_params 和 get_optimizer 是用于配置和创建优化器的函数。这些函数根据指定的网络类型为不同的模型组件设置参数组，并为其配置适当的学习率。最终，get_optimizer 函数根据这些参数组创建一个 Adam 优化器。下面是对每一行代码的详细解释：
def get_params(alg, args, nettype):  # get_params 函数  函数定义  定义函数：get_params 用于根据指定的网络类型 (nettype) 返回相应的模型参数组。  参数：alg：包含模型各组件的对象（通常是算法类的实例，如 Diversify）。args：包含各种配置参数的对象（如学习率、衰减系数等）。nettype：字符串，指定参数组的类型（如 'Diversify-adv'、'Diversify-cls'、'Diversify-all'）。
    init_lr = args.lr  # 初始化学习率  # 作用：获取初始学习率 init_lr，从 args 中读取。 原因：将学习率作为可配置参数，可以根据训练需要调整。
    if nettype == 'Diversify-adv':  # 设置参数组 - Diversify-adv   条件判断：检查 nettype 是否为 'Diversify-adv'。
        params = [  # 作用：为指定网络组件设置参数组。 包含瓶颈层 dbottleneck 的参数，并将学习率设置为 args.lr_decay2 * init_lr。
            {'params': alg.dbottleneck.parameters(), 'lr': args.lr_decay2 * init_lr},  # 作用：为指定网络组件设置参数组。 包含瓶颈层 dbottleneck 的参数，并将学习率设置为 args.lr_decay2 * init_lr。
            {'params': alg.dclassifier.parameters(), 'lr': args.lr_decay2 * init_lr},  # 包含域分类器 dclassifier 的参数，同样使用衰减后的学习率。
            {'params': alg.ddiscriminator.parameters(), 'lr': args.lr_decay2 * init_lr}  # 包含域判别器 ddiscriminator 的参数，使用相同的学习率设置。
        ]
        return params  # 原因：为特定的网络组件分配特定的学习率，有助于细粒度控制训练过程中的参数更新。
    elif nettype == 'Diversify-cls':  # 设置参数组 - Diversify-cls  条件判断：检查 nettype 是否为 'Diversify-cls'。 作用：为另一组网络组件设置参数组。
        params = [
            {'params': alg.bottleneck.parameters(), 'lr': args.lr_decay2 * init_lr},  # 包含瓶颈层 bottleneck 的参数，设置衰减后的学习率。
            {'params': alg.classifier.parameters(), 'lr': args.lr_decay2 * init_lr},  # 包含分类器 classifier 的参数，使用相同的学习率设置。
            {'params': alg.discriminator.parameters(), 'lr': args.lr_decay2 * init_lr}  # 包含判别器 discriminator 的参数，使用相同的学习率设置。
        ]  # 原因：根据网络类型选择不同的组件，分配合适的学习率，便于优化不同部分的网络。
        return params
    elif nettype == 'Diversify-all':  # 设置参数组 - Diversify-all  条件判断：检查 nettype 是否为 'Diversify-all'。  作用：为整个模型的所有组件设置参数组。
        params = [
            {'params': alg.featurizer.parameters(), 'lr': args.lr_decay1 * init_lr},  # 包含特征提取器 featurizer 的参数，设置不同的衰减学习率 args.lr_decay1 * init_lr。
            {'params': alg.abottleneck.parameters(), 'lr': args.lr_decay2 * init_lr},  # 包含另一个瓶颈层 abottleneck 的参数，使用第二种衰减学习率。
            {'params': alg.aclassifier.parameters(), 'lr': args.lr_decay2 * init_lr}  # 包含另一个分类器 aclassifier 的参数，使用相同的学习率设置。
        ]
        return params  # 原因：提供完整的参数设置，允许整个模型的所有部分都参与优化，并使用合适的学习率。


def get_optimizer(alg, args, nettype):  # 函数定义  定义函数：get_optimizer 用于创建优化器，根据指定的网络类型和参数组配置优化器。
    params = get_params(alg, args, nettype=nettype)  # alg：包含模型各组件的对象。 args：包含各种配置参数的对象。  nettype：指定网络类型，决定选择哪个参数组。nettype：指定网络类型，决定选择哪个参数组。 # 获取参数组 作用：调用 get_params 函数，获取特定网络类型的参数组。  原因：为优化器提供特定的参数组，以便配置优化策略。
    optimizer = torch.optim.Adam(  # 创建优化器  作用：创建 Adam 优化器 optimizer。  params：来自 get_params 的参数组。
        params, lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, 0.9)) #  lr=args.lr：学习率。  weight_decay=args.weight_decay：权重衰减，用于 L2 正则化。  betas=(args.beta1, 0.9)：一阶和二阶动量的系数。  原因：Adam 优化器通过结合动量和自适应学习率，能有效加速训练过程并稳定收敛。  # 原因：Adam 优化器通过结合动量和自适应学习率，能有效加速训练过程并稳定收敛。
    return optimizer    # 原因：Adam 优化器通过结合动量和自适应学习率，能有效加速训练过程并稳定收敛。
