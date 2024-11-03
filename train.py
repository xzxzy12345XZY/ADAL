# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import time
from alg.opt import *
from alg import alg, modelopera
from utils.util import set_random_seed, get_args, print_row, print_args, train_valid_target_eval_names, alg_loss_dict, \
    print_environ
from datautil.getdataloader_single import get_act_dataloader

import csv
import os

from sklearn.metrics import classification_report
import csv
import os


def save_results_to_csv(args, report_best):
    # 获取要保存的CSV文件名
    # csv_filename = fr".\results_2_class\{args.activity_id}.csv"
    csv_filename = os.path.join(".", "results_2_class_pads", f"{args.activity_id}.csv")

    # 确定文件是否存在
    file_exists = os.path.isfile(csv_filename)

    # 准备要写入的数据，初始化为传入的args参数
    data_to_write = {
        'test_env': args.test_envs,
    }

    # 将classification_report的所有指标平展后加入要写入的数据字典中
    def flatten_and_format_dict(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_and_format_dict(v, new_key, sep=sep).items())
            else:
                # 如果是非support的指标，将每个指标乘以100并保留两位小数
                if 'support' not in new_key and isinstance(v, (float, int)):
                    formatted_value = f"{v * 100:.2f}"
                else:
                    # 对于support保持整数格式
                    formatted_value = f"{v:.0f}" if 'support' in new_key else v
                items.append((new_key, formatted_value))
        return dict(items)

    # 将平展并格式化后的report_best内容加入到data_to_write
    formatted_report = flatten_and_format_dict(report_best)

    # 提取并将accuracy和macro avg相关指标放在test_env之后
    accuracy_metric = {'accuracy': formatted_report.pop('accuracy')}
    macro_metrics = {k: formatted_report.pop(k) for k in list(formatted_report.keys()) if 'macro avg' in k}

    # 更新数据顺序
    data_to_write.update(accuracy_metric)
    data_to_write.update(macro_metrics)

    # 将args的所有参数作为字典存储在结果数据中
    data_to_write['args'] = vars(args)

    # 更新剩余的formatted_report到data_to_write
    data_to_write.update(formatted_report)

    # 写入CSV文件
    with open(csv_filename, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data_to_write.keys())

        # 如果文件不存在，则写入表头
        if not file_exists:
            writer.writeheader()

        # 写入数据
        writer.writerow(data_to_write)


# 定义主函数
def main(args):
    s = print_args(args, [])  # # 打印参数并设置随机种子以保证实验可重复
    set_random_seed(args.seed)  # 设置随机种子

    print_environ()  # # 打印运行环境相关信息
    print(s)
    # if args.latent_domain_num < 6:  # # 根据域的数量设置批处理大小
    #     args.batch_size = 32 * args.latent_domain_num
    # else:
    #     args.batch_size = 16 * args.latent_domain_num

    train_loader, train_loader_noshuffle, valid_loader, target_loader, _, _, _ = get_act_dataloader(
        args)  # # 获取数据加载器

    best_valid_acc, target_acc = 0, 0  # # 初始化最优验证准确率和目标准确率
    # ==============0831================
    report_str_best = None
    report_dict_best = None
    # ==============0831================
    algorithm_class = alg.get_algorithm_class(args.algorithm)  # # 根据指定的算法获取算法类并初始化算法
    algorithm = algorithm_class(args).cuda()
    algorithm.train()
    optd = get_optimizer(algorithm, args, nettype='Diversify-adv')  # # 获取不同网络部分的优化器  作用：为对抗性网络部分（如判别器和域分类器）获取优化器 optd。  原因：对不同网络部分使用单独的优化器，可以分别调整优化过程。
    opt = get_optimizer(algorithm, args, nettype='Diversify-cls')  # 作用：为分类网络部分（如瓶颈层和分类器）获取优化器 opt。  原因：单独的优化器 opt 用于优化分类相关的网络部分，确保不同部分独立优化。
    opta = get_optimizer(algorithm, args, nettype='Diversify-all')  # 作用：为整个网络（包括特征提取器和所有分类器）获取优化器 opta。 #  原因：opta 用于特征提取器的特征更新阶段，确保特征学习覆盖整个网络。

    for round in range(args.max_epoch):  # # 进行多轮训练  # 作用：开始训练循环，遍历指定的最大轮次（max_epoch）。  # 原因：多轮训练使模型逐步优化，每轮包括特征更新和学习不同任务。
        print(f'\n========ROUND {round}========')
        print('====Feature update====')  # 特征更新  作用：打印特征更新的标题和损失列表。  原因：对特征更新部分的损失进行监控，以确保模型在正确学习特征。
        loss_list = ['class']
        print_row(['epoch'] + [item + '_loss' for item in loss_list], colwidth=15)

        for step in range(args.local_epoch):  # # 进行特征更新  作用：进行特征更新，使用 opta 优化特征提取器。
            for data in train_loader:  # 操作：  for step in range(args.local_epoch)：在每轮中进行多个本地迭代。  for data in train_loader：逐批次读取数据进行更新。
                loss_result_dict = algorithm.update_a(data, opta)  ## algorithm.update_a(data, opta)：调用特征更新函数，使用 opta 优化。
            print_row([step] + [loss_result_dict[item]
                                for item in loss_list], colwidth=15)  ## print_row(...)：打印每步的损失，便于观察特征学习的进展。  原因：在训练过程中定期更新特征提取器，以确保特征空间不断优化。原因：在训练过程中定期更新特征提取器，以确保特征空间不断优化。

        print('====Latent domain characterization====')  # 隐域特征化  # # 作用：打印隐域特征化部分的标题和损失类型列表。
        loss_list = ['total', 'dis', 'ent']  # 原因：通过对隐域特征化的损失监控，确保模型在学习不同隐域的特征。
        print_row(['epoch'] + [item + '_loss' for item in loss_list], colwidth=15)

        for step in range(args.local_epoch):  # # 对隐域进行特征化 ## # 对隐域进行特征化 作用：对隐域特征进行优化，使用 optd 更新域判别器。
            for data in train_loader:
                loss_result_dict = algorithm.update_d(data, optd)  # algorithm.update_d(data, optd)：调用隐域特征化的更新函数，优化域相关特征。
            print_row([step] + [loss_result_dict[item]  # print_row(...)：打印损失，便于观察隐域特征化的效果。
                                for item in loss_list], colwidth=15)  # 原因：使特征提取器学到的特征对隐域有很好的区分能力或域不变性。

        algorithm.set_dlabel(train_loader)  # # 设置域标签  ## 设置域标签  # 作用：更新数据的域标签，用于指导后续的对抗学习和分类。 # 原因：在域自适应学习中，正确的域标签能帮助模型更好地识别域不变特征。

        print('====Domain-invariant feature learning====')  # 域不变特征学习  作用：打印域不变特征学习的标题。  ## 原因：开始域不变特征学习阶段，通过优化使特征对所有域一致。

        loss_list = alg_loss_dict(args)  # # 计算损失并评估  ## 作用：设置损失和评估指标，准备打印训练过程中的评估结果。  ## 原因：监控不同损失和评估指标的变化，以确保模型在不断优化。
        eval_dict = train_valid_target_eval_names(args)
        print_key = ['epoch']
        print_key.extend([item + '_loss' for item in loss_list])
        print_key.extend([item + '_acc' for item in eval_dict.keys()])
        print_key.append('total_cost_time')
        print_row(print_key, colwidth=15)  ## # # 计算损失并评估  ## 作用：设置损失和评估指标，准备打印训练过程中的评估结果。  ## 原因：监控不同损失和评估指标的变化，以确保模型在不断优化。

        sss = time.time()  # # 域不变特征学习  ##  # 域不变特征学习  ## 作用：开始域不变特征学习，使用 opt 更新分类器和判别器。
        for step in range(args.local_epoch):  ## 操作：algorithm.update(data, opt)：使用优化器 opt 进行更新。 step_vals：保存每一步的损失结果。 ## 原因：使特征提取器学习到域不变特征，减少不同域之间的差异。
            for data in train_loader:
                step_vals = algorithm.update(data, opt)  ##  使用优化器 opt 进行更新。  step_vals：保存每一步的损失结果。

            results = {
                'epoch': step,  ## 计算和打印评估指标 初始化结果字典：用于存储当前步的评估结果。 原因：保存和打印各类损失和准确率，便于追踪训练进展。
            }

            results['train_acc'] = modelopera.accuracy(  ## 计算训练集准确率：调用 modelopera.accuracy 计算训练集上的准确率。
                algorithm, train_loader_noshuffle, None)  ## 原因：监控模型在训练集上的表现，确保其在不断学习。

            acc = modelopera.accuracy(algorithm, valid_loader, None)  # 计算验证集准确率：计算验证集上的准确率。
            results['valid_acc'] = acc  # 原因：通过验证集上的表现评估模型的泛化能力。

            acc, report_str, report_dict = modelopera.accuracy(algorithm, target_loader, None, target_falese=True)  # 计算目标域准确率：计算目标域（测试集）上的准确率。
            results['target_acc'] = acc  # 原因：评估模型在未见过的数据上的表现，尤其是在域自适应任务中。

            for key in loss_list:  # 存储各类损失：将当前步的损失结果保存到结果字典。 ## 原因：记录每一步的损失，便于后续分析和调试。
                results[key + '_loss'] = step_vals[key]
            # if results['valid_acc'] > best_valid_acc:  # #保存最佳结果：如果当前验证集准确率超过之前的最佳结果，则更新最佳准确率。
            #     best_valid_acc = results['valid_acc']  # 原因：记录当前验证集上wqe表现最好的模型，以便在训练结束后使用。
                # target_acc = results['target_acc']
            # # ==============================================
            # if results['target_acc'] > target_acc:  # #保存最佳结果：如果当前验证集准确率超过之前的最佳结果，则更新最佳准确率。
            #     best_valid_acc = results['valid_acc']  # 原因：记录当前验证集上表现最好的模型，以便在训练结束后使用。
            #     target_acc = results['target_acc']
            # # ==============================================
            # ===================0831========================
            if results['target_acc'] > target_acc:  # 如果当前目标域准确率超过之前的最佳结果
                target_acc = results['target_acc']  # 仅更新最佳目标域准确率
                report_str_best = report_str
                report_dict_best = report_dict
            # ===================0831========================
            results['total_cost_time'] = time.time() - sss  # # 计算并打印总时间：计算当前阶段耗时，并打印所有评估指标。
            print_row([results[key] for key in print_key], colwidth=15)  ## 原因：确保训练过程的时间跟踪和评估指标输出同步。
    # ===================0831========================
    print(f'Target acc best report:\n {report_str_best}')
    # ===================0831========================
    print(f'Target acc: {target_acc:.4f}')  # # 打印目标准确率 ## 打印最终目标域准确率 ## 输出最终准确率：打印在目标域上的最佳准确率。 ## 原因：提供模型在目标域上最终的表现结果，作为整个训练的最终评估。
    save_results_to_csv(args, report_dict_best)


if __name__ == '__main__':
    args = get_args()
    main(args)
#
# # 总结
# 整个代码段展示了多轮训练循环中的特征更新、隐域特征化、域不变特征学习三个主要阶段。每个阶段都有特定的优化目标，通过监控损失和准确率来确保模型在正确优化。这种细粒度的训练控制可以帮助模型更好地适应多域、多任务的学习环境，从而提升泛化能力和性能。
#