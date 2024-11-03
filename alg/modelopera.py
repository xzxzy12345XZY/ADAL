# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from network import act_network


def get_fea(args):
    net = act_network.ActNetwork(args.dataset)
    return net


def accuracy(network, loader, weights, usedpredict='p', target_falese=False):  # 作用: 定义一个名为 accuracy 的函数，用于计算模型的准确率。 ## 参数: network: 神经网络模型实例，具有 predict 和 predict1 方法。 loader: 数据加载器（DataLoader 对象），用于提供批次数据。 weights: 样本权重（可以为 None），用于加权准确率计算。 usedpredict: 指定使用哪种预测方法（'p' 表示使用 predict 方法，其他值使用predict1 方法）。
    correct = 0  # correct: 累计计算模型的正确预测数量。
    total = 0  # total: 累计计算权重总和，用于归一化计算加权准确率。
    weights_offset = 0  # weights_offset: 用于在权重数组 weights 中跟踪当前批次的权重起始位置。

    # ======0831======== #
    # 初始化混淆矩阵
    all_preds = []
    all_labels = []
    # ======0831======== #

    network.eval()  # 作用: 将模型设置为评估模式。  原因: 在评估模式下，模型将禁用 Dropout 和 BatchNorm 等操作，从而确保推断过程的稳定性。
    with torch.no_grad():  # 作用: 禁用梯度计算。
        for data in loader:  # 原因: 在评估模式下不需要计算梯度，这样可以减少内存消耗并加速计算。 作用: 迭代加载器中的数据。
            x = data[0].cuda().float()  # x: 输入数据（特征），将其转换为浮点类型，并移动到 GPU。
            y = data[1].cuda().long()  # y: 目标标签，将其转换为长整型，并移动到 GPU。 原因: 通过 GPU 加速计算，并确保数据类型与模型兼容。
            if usedpredict == 'p':  # 作用: 根据 usedpredict 的值选择使用哪种预测方法。  如果 usedpredict 为 'p'，则使用 network.predict 方法。  否则，使用 network.predict1 方法。 原因: 支持在同一函数中使用不同的预测逻辑，便于灵活地评估模型的不同部分或策略。
                p = network.predict(x)
            else:
                p = network.predict1(x)
            if weights is None:  #  作用: 计算当前批次的样本权重。
                batch_weights = torch.ones(len(x))  # 如果 weights 为 None，则为每个样本分配权重 1。
            else:  # 如果 weights 不为 None，则从权重数组中提取当前批次的权重，并更新 weights_offset 以供下次使用。 原因: 支持加权计算准确率，以便根据不同样本的重要性进行准确率的加权平均。
                batch_weights = weights[weights_offset:
                                        weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.cuda()  # 作用: 将权重移动到 GPU   ,原因: 保证计算时数据在相同的设备（GPU）上，从而避免因数据在不同设备上导致的错误或性能问题。
            if p.size(1) == 1:  # 作用: 计算当前批次的正确预测数，并加权累计到 correct。 # 如果 p 的第二维大小为 1，表示二分类任务，使用 p.gt(0)（预测值大于 0）判断正类，
                # ==========0831==============
                raise ValueError('这里可不是二分类任务，或者预测任务')
                preds = p.gt(0).long().view(-1)
                # ==========0831==============
                # ============0904==================
                # correct += (p.gt(0).eq(y).float() *
                #             batch_weights.view(-1, 1)).sum().item()  # eq(y) 检查与真实标签 y 的匹配。 p.gt(0).eq(y).float() * batch_weights.view(-1, 1)：将结果转换为浮点型并乘以批次权重，然后计算总和。
                # ============0904==================

            else:
                # ==========0831==============
                preds = p.argmax(1)
                # ==========0831==============
                # ========= 0904==================
                # correct += (p.argmax(1).eq(y).float() *
                #             batch_weights).sum().item()  # 如果 p 的第二维大小大于 1，表示多分类任务，使用 p.argmax(1) 获取预测的类别索引，eq(y) 检查预测是否与真实标签 y 相等。 p.argmax(1).eq(y).float() * batch_weights：将结果转换为浮点型并乘以批次权重，然后计算总和。 原因: 为不同任务类型（例如二分类和多分类）计算准确率。
                # ========= 0904==================
                preds = preds.view(-1, 18)
                y = y.view(-1, 18)
                # 使用 torch.mode 获取每个人预测的众数（出现次数最多的标签）
                final_preds, _ = torch.mode(preds, dim=1)
                final_y, _ = torch.mode(y, dim=1)
                # 将当前批次的预测和真实标签添加到列表中
                all_preds.extend(final_preds.cpu().numpy())
                all_labels.extend(final_y.cpu().numpy())
            # total += batch_weights.sum().item()  ## 作用: 累加批次样本权重总和到 total。 原因: 用于计算加权准确率的分母。
            # ==========0831==============
            # 将当前批次的预测和真实标签添加到列表中
            # all_preds.extend(preds.cpu().numpy())
            # all_labels.extend(y.cpu().numpy())
            # ==========0831==============
            # # ==========0831==============
            # preds = preds.view(-1, 36)
            # y = y.view(-1, 36)
            # # 使用 torch.mode 获取每个人预测的众数（出现次数最多的标签）
            # final_preds, _ = torch.mode(preds, dim=1)
            # final_y, _ = torch.mode(y, dim=1)
            # # 将当前批次的预测和真实标签添加到列表中
            # all_preds.extend(final_preds.cpu().numpy())
            # all_labels.extend(final_y.cpu().numpy())
            # # ==========0831==============

    # ==========0831==============
    if target_falese:
        # 计算混淆矩阵
        report_str = classification_report(all_labels, all_preds, digits=4, zero_division=0)
        report_dict = classification_report(all_labels, all_preds, output_dict=True, digits=4, zero_division=0)
        # print("TargetDataLoader Confusion Matrix:")
        # print(report)
        network.train()  # 作用: 将模型恢复到训练模式。  原因: 恢复 Dropout 和 BatchNorm 等操作，确保模型在后续训练时的行为正常。
        # return correct / total, report
        return accuracy_score(all_labels, all_preds), report_str, report_dict

    # ==========0831==============
    network.train()  # 作用: 将模型恢复到训练模式。  原因: 恢复 Dropout 和 BatchNorm 等操作，确保模型在后续训练时的行为正常。

    # return correct / total  # 原因: 恢复 Dropout 和 BatchNorm 等操作，确保模型在后续训练时的行为正常。  ## 原因: 用于评估模型的性能表现，尤其是考虑不同样本权重情况下的准确率。
    return accuracy_score(all_labels, all_preds)  # 原因: 恢复 Dropout 和 BatchNorm 等操作，确保模型在后续训练时的行为正常。  ## 原因: 用于评估模型的性能表现，尤其是考虑不同样本权重情况下的准确率。

#
#
# 总结
# 这段代码通过遍历测试数据集中的每个批次，计算每个批次的加权准确率，并最终返回整体的加权准确率结果。它灵活地支持不同的预测方法、不同的任务类型（如二分类和多分类），以及加权和不加权的准确率计算。
#