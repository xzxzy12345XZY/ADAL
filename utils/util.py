# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import json
import random
import numpy as np
import pandas as pd
import torch
import sys
import os
import argparse
import torchvision
import PIL


def set_random_seed(seed=0):  # # 设置随机种子以确保结果可重复 ## 定义函数：set_random_seed 用于设置随机种子，确保实验的可重复性。
    random.seed(seed)  # seed=0：默认随机种子为 0，可以更改为其他整数值。 ##作用：设置 Python 内置的 random 模块的随机种子。 ## 原因：控制所有使用 random 生成的随机数，使其输出一致。
    np.random.seed(seed)  ## 作用：设置 NumPy 的随机数种子。  ## 原因：控制 NumPy 中生成的随机数，使结果可重复。
    torch.manual_seed(seed)  ## 作用：设置 PyTorch CPU 上的随机种子。 ## 原因：确保在 PyTorch 中所有 CPU 计算（如初始化权重）是可重复的。
    torch.cuda.manual_seed(seed)  ## 作用：设置 PyTorch GPU 上的随机种子。 ## 原因：确保在 PyTorch 中所有 GPU 计算是可重复的，特别是涉及 CUDA 操作的。
    torch.backends.cudnn.deterministic = True  ## 作用：设置 cuDNN 后端为确定性模式。  ## 原因：启用确定性操作，以消除可能的随机性，保证结果的一致性。
    torch.backends.cudnn.benchmark = False  ## 作用：禁用 cuDNN 的自动优化。  ## 原因：防止 cuDNN 对不同输入尺寸使用不同算法，从而引入不可控的随机性。


def train_valid_target_eval_names(args):  # # 根据args中的域数量和测试环境，定义评估指标的名称  ## 原因：防止 cuDNN 对不同输入尺寸使用不同算法，从而引入不可控的随机性。  ##  args：包含域数量和测试环境等参数的对象。
    eval_name_dict = {'train': [], 'valid': [], 'target': []}  ## 作用：初始化一个字典，用于存储不同类型的评估指标名称。  ## 原因：区分训练、验证和目标测试环境的评估名称，有助于后续的结果整理和输出。
    for i in range(args.domain_num):  # 作用：遍历所有域的编号。 # 原因：生成每个域的评估指标名称。
        if i not in args.test_envs:  ## 条件判断：检查当前域是否不在测试环境中。
            eval_name_dict['train'].append('eval%d_in' % i)  ## eval_name_dict['train'].append('eval%d_in' % i)：将当前域的训练评估名称添加到训练列表。
            eval_name_dict['valid'].append('eval%d_out' % i) ## eval_name_dict['valid'].append('eval%d_out' % i)：将当前域的验证评估名称添加到验证列表。
        else:
            eval_name_dict['target'].append('eval%d_out' % i)  # 作用：如果当前域在测试环境中，将其评估名称添加到目标测试列表。 # 原因：标记目标域的评估名称，便于在训练过程中对目标域的表现进行监控。
    return eval_name_dict  # 返回结果：返回包含训练、验证和目标测试环境的评估名称字典。  # 原因：输出评估名称字典，以便在训练和评估过程中使用。


def alg_loss_dict(args):  # # 根据算法类型返回特定的损失类型列表  ##   # 根据算法类型返回特定的损失类型列表  # 定义函数：alg_loss_dict 根据给定的算法类型返回对应的损失类型列表。#args：包含算法类型的参数对象。
    loss_dict = {'diversify': ['class', 'dis', 'total']}  ## 定义损失字典：包含不同算法类型及其对应的损失类型。  # 原因：根据算法类型，返回特定的损失名称，便于训练过程中选择合适的损失计算。
    return loss_dict[args.algorithm]  ## 返回结果：根据 args.algorithm 返回对应的损失类型列表。 ## 原因：确保根据当前使用的算法类型，正确选择损失计算方法。


def print_args(args, print_list):  # # 打印参数  ## print_args 函数 ## args：包含所有参数的对象。  ## print_list：要打印的参数列表，如果为空，则打印所有参数。
    s = "==========================================\n"  ## 作用：初始化一个字符串，用于存储打印输出的内容。  ## 原因：作为分隔符，增强输出的可读性。
    l = len(print_list)  ## 作用：获取 print_list 的长度。  ## 原因：用于判断是否打印所有参数或仅打印特定参数。
    for arg, content in args.__dict__.items():  ## 遍历参数：遍历 args 中的所有参数和值。
        if l == 0 or arg in print_list:  ## if l == 0 or arg in print_list：如果 print_list 为空或参数在 print_list 中，则将参数和值添加到字符串 s。 ## 原因：根据 print_list 打印指定参数，或打印所有参数。
            s += "{}:{}\n".format(arg, content)
    return s  ## 返回结果：返回包含参数和其值的字符串。


def print_row(row, colwidth=10, latex=False):  # # 打印行信息，支持普通和LaTeX格式 ## print_row 函数 ## 定义函数：print_row 用于打印一行信息，支持普通和 LaTeX 格式。 # row：要打印的行信息（列表）。 # colwidth=10：每列的宽度，默认为 10。 ## latex=False：是否使用 LaTeX 格式打印。
    if latex:  ## 条件判断：检查是否使用 LaTeX 格式。 ## 设置分隔符和结束符：
        sep = " & "  ##  sep = " & "：LaTeX 格式中使用 & 分隔列。
        end_ = "\\\\"  ## end_ = "\\\\"：LaTeX 格式中使用 \\ 作为行结束符。
    else:
        sep = "  "  ## sep = " "：列之间用两个空格分隔。
        end_ = ""  ## end_ = ""：行尾不加特殊符号。  ## 原因：根据用户的选择，输出普通或 LaTeX 格式。

    def format_val(x):  ## 定义内部函数 format_val：用于格式化值。## x：要格式化的值。
        if np.issubdtype(type(x), np.floating):  ## 操作： 检查类型：如果 x 是浮点数类型，则格式化为小数点后十位。
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]  ## 调整列宽：使用 ljust 方法调整列宽到指定宽度。 原因：确保输出对齐且格式一致。
    print(sep.join([format_val(x) for x in row]), end_)  ## 作用：打印格式化后的行信息。 ## 原因：输出格式化的行信息，以便观察和分析。


def print_environ():  # # 打印当前Python和相关库的版本信息 ## print_environ 函数 ## 定义函数：print_environ 用于打印当前 Python 环境及相关库的版本信息。
    print("Environment:")  ## 打印标题：打印 "Environment:" 作为环境信息的标题。  ## 原因：标识输出内容是环境信息。
    print("\tPython: {}".format(sys.version.split(" ")[0]))  ## 打印 Python 版本：通过 sys.version.split(" ")[0] 获取 Python 的主版本号并打印。  ## 原因：显示 Python 版本，便于追踪运行环境。
    print("\tPyTorch: {}".format(torch.__version__))  ## 打印 PyTorch 版本：通过 torch.__version__ 获取 PyTorch 版本并打印。 ## 原因：确认 PyTorch 版本，确保代码与环境匹配。
    print("\tTorchvision: {}".format(torchvision.__version__))  ## 打印 Torchvision 版本：获取并显示 Torchvision 版本。 ## 原因：显示图像相关操作库版本，确保兼容性。
    print("\tCUDA: {}".format(torch.version.cuda))  ## 打印 CUDA 版本：通过 torch.version.cuda 获取 CUDA 版本并打印。  ## 原因：显示 CUDA 版本，以便确认 GPU 加速的环境设置。
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))  ## 打印 cuDNN 版本：获取 cuDNN 版本，显示深度学习加速库的版本信息。  ## 原因：验证 cuDNN 版本，确保深度学习操作的性能和兼容性。
    print("\tNumPy: {}".format(np.__version__))  ## 打印 NumPy 版本：通过 np.__version__ 显示 NumPy 版本。 ##
    print("\tPIL: {}".format(PIL.__version__))  ## 打印 PIL 版本：通过 PIL.__version__ 显示 Python 图像库版本。 ## 原因：确认图像处理库的版本信息。

# 定义类：Tee 类用于重定向标准输出和错误输出到指定文件的同时保留输出到控制台的功能。 目的：记录程序的输出到文件中，同时显示在控制台，这样便于调试和日志记录。
class Tee:  # # 重定向标准输出和错误到文件和控制台  Tee 类用于同时将输出重定向到控制台和文件，类似于 Unix 命令 tee 的功能。它通过拦截标准输出和错误，将消息写入文件的同时保持原有的输出到控制台的行为。这对于在运行程序时记录日志非常有用。以下是对 Tee 类及其每一行代码的详细解释：
    def __init__(self, fname, mode="a"):  # 初始化方法：定义 __init__ 方法，用于创建 Tee 类的实例。 fname：文件名，用于指定输出的日志文件。 mode="a"：文件打开模式，默认是追加模式 "a"，可以写入新内容而不覆盖已有内容。
        self.stdout = sys.stdout  # 保存当前标准输出：将当前的标准输出（通常是控制台）保存到 self.stdout。  原因：保存原始标准输出的引用，便于在重定向输出的同时保持原有功能。
        self.file = open(fname, mode)  # 打开文件：根据指定的文件名 fname 和模式 mode 打开文件，将文件对象保存到 self.file。  原因：打开一个文件以供写入输出信息，模式通常为 "a"（追加模式），这样不会覆盖已有的日志。

    def write(self, message):  # 定义 write 方法：用于将输出信息同时写入控制台和文件。  message：要写入的字符串信息（如标准输出的内容）。
        self.stdout.write(message)  # 输出到控制台：将消息写入保存的原始标准输出 self.stdout（即控制台）。  原因：保持输出到控制台的行为，让用户在运行时仍能看到程序的输出。
        self.file.write(message)  # 输出到文件：将消息写入打开的文件对象 self.file。  原因：将输出保存到文件中，用于日志记录，这样便于以后查看或分析输出内容。
        self.flush()  # 刷新输出：调用 flush() 方法，确保输出缓冲区的内容立即写入目标位置（控制台和文件）。  # 原因：强制

    def flush(self):  # 定义 flush 方法：用于刷新输出缓冲区。  目的：确保将所有输出立即写入，不等待缓冲区满或程序结束。
        self.stdout.flush()  # 刷新控制台输出：调用原始标准输出的 flush() 方法，将控制台缓冲区的内容立即输出。
        self.file.flush()  # 原因：确保控制台显示最新的输出信息，防止数据滞留在缓冲区中。


def act_param_init(args):  # # 设置活动识别参数
    # args.select_position = {'emg': [0]}  # 选择位置设置:  这行代码在 args 对象中设置一个字典，指定电肌图（EMG）信号的选择位置。这里的 [0] 可能表示选择第一个位置的信号，具体含义取决于数据集的结构。
    # args.select_channel = {'emg': np.arange(8)}  # args.select_channel = {'emg': np.arange(8)} 这行代码为EMG数据设置选择的频道。np.arange(8) 生成一个数组 [0, 1, 2, 3, 4, 5, 6, 7]，表示选择前8个频道的数据。
    # args.hz_list = {'emg': 1000}  # 设置赫兹（Hz）列表: 这行代码设置一个字典，指定EMG信号的采样率为1000Hz。这是电肌图信号处理中的一个常见设置，用于确定数据处理和特征提取的时间分辨率。
    # args.act_people = {'emg': [[i*9+j for j in range(9)]for i in range(4)]}  # 活动人员设置:  这行代码构造了一个嵌套列表，它为EMG数据定义了涉及的人员。列表解析中 i*9+j for j in range(9) 为每组4人创建了9个序号，可能表示4组人员的不同配置或实验条件。
    # tmp = {'emg': ((8, 1, 200), 6, 10)}  # 临时变量定义和参数分配: (8, 1, 200): 数据的输入形状。 6: 类别数。 10: 网格大小。
    # args.num_classes, args.input_shape, args.grid_size = tmp[
    #     args.dataset][1], tmp[args.dataset][0], tmp[args.dataset][2]  # 这些值随后被分配给 args.num_classes（类别数），args.input_shape（输入形状），和 args.grid_size（网格大小）。
    # ====================0830====================================
    args.select_position = {
        'pads': [0]}  # 选择位置设置:  这行代码在 args 对象中设置一个字典，指定电肌图（EMG）信号的选择位置。这里的 [0] 可能表示选择第一个位置的信号，具体含义取决于数据集的结构。
    args.select_channel = {'pads': np.arange(
        6)}  # args.select_channel = {'emg': np.arange(8)} 这行代码为EMG数据设置选择的频道。np.arange(8) 生成一个数组 [0, 1, 2, 3, 4, 5, 6, 7]，表示选择前8个频道的数据。
    args.hz_list = {'pads': 100}  # 设置赫兹（Hz）列表: 这行代码设置一个字典，指定EMG信号的采样率为1000Hz。这是电肌图信号处理中的一个常见设置，用于确定数据处理和特征提取的时间分辨率。
    # args.act_people = {'emg': [[i * 9 + j for j in range(9)] for i in range(
    #     4)]}  # 活动人员设置:  这行代码构造了一个嵌套列表，它为EMG数据定义了涉及的人员。列表解析中 i*9+j for j in range(9) 为每组4人创建了9个序号，可能表示4组人员的不同配置或实验条件。
    # 读取CSV文件
    # loaded_df = pd.read_csv('subject_grouping.csv')
    # # 从JSON格式还原为Python字典
    # restored_groups = json.loads(loaded_df['groups'].iloc[-1])
    loaded_df = pd.read_csv('subject_grouping_pads_2_class.csv')
    # # 从JSON格式还原为Python字典
    # restored_groups = json.loads(loaded_df['groups'].iloc[-1])
    # # 将还原的分组结果展示为Python列表格式
    # restored_lists = {int(k): v for k, v in restored_groups.items()}
    # 根据Activity_id匹配正确的分组结果
    print("当前的args.activity_id:" + str(args.activity_id))
    matched_row = loaded_df[loaded_df['dataset'] == args.activity_id]
    if not matched_row.empty:
        # 从JSON格式还原为Python字典
        restored_groups = json.loads(matched_row['groups'].iloc[-1])

        # 将还原的分组结果展示为Python列表格式
        restored_lists = {int(k): v for k, v in restored_groups.items()}
        print('还原的分组结果:', restored_lists)
    else:
        raise ValueError(f'未找到Activity_id为 {args.activity_id} 的分组结果')
        # print(f'未找到Activity_id为 {args.activity_id} 的分组结果')
    # 将还原的分组结果展示为Python列表格式
    restored_lists = {int(k): v for k, v in restored_groups.items()}
    args.act_people = {'pads': [restored_lists[i] for i in range(
        4)]}
    tmp = {'pads': ((6, 1, 100), 2, 10)}  # 临时变量定义和参数分配: (6, 1, 200): 数据的输入形状。 4: 类别数。 10: 网格大小。
    args.num_classes, args.input_shape, args.grid_size = tmp[
        args.dataset][1], tmp[args.dataset][0], tmp[args.dataset][2]
    # ====================0830====================================
    return args


def get_args(custom_args=None):  # # 解析命令行参数
    parser = argparse.ArgumentParser(description='DG')
    parser.add_argument('--algorithm', type=str, default="diversify")  # --algorithm: 设置使用的算法，默认为 "diversify"。这通常指定了底层模型或训练方法。
    parser.add_argument('--alpha', type=float,
                        default=0.1, help="DANN dis alpha")  # --alpha: 设置DANN（Domain-Adversarial Neural Network）中的领域判别器损失函数的权重，默认为 0.1。
    # parser.add_argument('--alpha1', type=float,
    #                     default=0.1, help="DANN dis alpha")
    parser.add_argument('--alpha1', type=float,
                        default=1.0, help="DANN dis alpha")  # --alpha1: 同样用于调节DANN中的某个参数，这里默认为1.0，覆盖了之前的默认值。
    parser.add_argument('--batch_size', type=int,
                        default=18*64, help="batch_size")  # --batch_size: 设置每个训练批次中的样本数，默认为32。
    parser.add_argument('--beta1', type=float, default=0.5, help="Adam")  # --beta1: 设置Adam优化器中的beta1参数，默认为0.5，影响一阶矩估计的指数衰减率。
    parser.add_argument('--bottleneck', type=int, default=256)  # --bottleneck: 设置模型中瓶颈层的神经元数量，默认为256。
    parser.add_argument('--checkpoint_freq', type=int,
                        default=100, help='Checkpoint every N steps')  # --checkpoint_freq: 设置每多少训练步骤保存一次模型，默认为每100步。
    parser.add_argument('--classifier', type=str,
                        default="linear", choices=["linear", "wn"])  # --classifier: 选择分类器类型，可选 "linear" 或 "wn"（可能指带权重归一化的网络），默认为 "linear"。
    parser.add_argument('--data_file', type=str, default='')  # --data_file: 设置数据文件的路径，默认为空。
    # parser.add_argument('--dataset', type=str, default='dsads')
    # ============================0830==========================================
    # parser.add_argument('--dataset', type=str, default='emg')
    parser.add_argument('--dataset', type=str, default='pads')
    # ============================0830==========================================
    # parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='./data/')  # --data_dir: 设置数据目录的路径，默认为 './data/'。如果同时指定 data_file 和 data_dir，两者将被连接使用。
    parser.add_argument('--dis_hidden', type=int, default=256)  # --dis_hidden: 设置某种网络（可能是判别器）的隐藏层大小，默认为256。
    parser.add_argument('--gpu_id', type=str, nargs='?',
                        default='0', help="device id to run")  # --gpu_id: 设置用于训练的GPU编号，默认为 '0'。
    parser.add_argument('--layer', type=str, default="bn",
                        choices=["ori", "bn"])  # --layer: 设置网络层的类型，可选择 "ori"（原始）或 "bn"（批归一化），默认为 "bn"。
    parser.add_argument('--lam', type=float, default=0.0)  # --lam: 可能用于调整正则化或其他损失组件的权重，默认为0.0。
    # parser.add_argument('--latent_domain_num', type=int, default=3)
    parser.add_argument('--latent_domain_num', type=int, default=8)  # --latent_domain_num: 设置潜在域的数量，默认为10。
    # parser.add_argument('--local_epoch', type=int,
    #                     default=1, help='local iterations')
    parser.add_argument('--local_epoch', type=int,
                        default=10, help='local iterations')  # --local_epoch: 设置每轮全局训练中的本地训练迭代次数，默认为3。
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")  # --lr: 设置学习率，默认为0.01。
    parser.add_argument('--lr_decay1', type=float,
                        default=1.0, help='for pretrained featurizer')  # --lr_decay1 和 --lr_decay2: 设置学习率衰减参数，默认都为1.0。
    parser.add_argument('--lr_decay2', type=float, default=1.0)  # --lr_decay1 和 --lr_decay2: 设置学习率衰减参数，默认都为1.0。
    # parser.add_argument('--max_epoch', type=int,
    #                     default=120, help="max iterations")
    parser.add_argument('--max_epoch', type=int,
                        default=100, help="max iterations")  # --max_epoch: 设置最大训练轮数，默认为50。
    parser.add_argument('--model_size', default='median',
                        choices=['small', 'median', 'large', 'transformer'])  # --model_size: 设置模型大小，选项包括 'small', 'median', 'large', 'transformer'，默认为 'median'。
    parser.add_argument('--N_WORKERS', type=int, default=4)  # --N_WORKERS: 设置加载数据时使用的工作进程数，默认为4。
    parser.add_argument('--old', action='store_true')  # --old: 一个标志位，用于启用一些旧的行为或配置。
    parser.add_argument('--seed', type=int, default=0)  # --seed: 设置随机种子以确保可重复性，默认为0。
    parser.add_argument('--task', type=str, default="cross_people")  # --task: 设置任务类型，默认为 "cross_people"，可能指跨个体的任务。
    parser.add_argument('--test_envs', type=int, nargs='+', default=[1])  # --test_envs: 设置测试环境的编号列表，用于评估，这里默认为列表 [0]。
    # parser.add_argument('--output', type=str, default="train_output")  # -output: 设置输出目录的路径，默认为 "train_output"。
    # parser.add_argument('--output', type=str, default="./data/train_output/act/cross_people-emg-Diversify-0-10-1-1-0-3-50-0.01")  # -output: 设置输出目录的路径，默认为 "train_output"。
    parser.add_argument('--output', type=str, default="./data/train_output/act/cross_people-our-Diversify-0-10-1-1-0-3-50-0.01")  # -output: 设置输出目录的路径，默认为 "train_output"。
    # parser.add_argument('--activity_id', type=int, default=10)  # --weight_decay: 设置权重衰减（正则化参数），默认为0.0005。
    # parser.add_argument('--weight_decay', type=float, default=5e-4)  # --weight_decay: 设置权重衰减（正则化参数），默认为0.0005。
    parser.add_argument('--weight_decay', type=float, default=5e-4)  # --weight_decay: 设置权重衰减（正则化参数），默认为0.0005。
    parser.add_argument('--activity_id', type=int, default=5)  # --weight_decay: 设置权重衰减（正则化参数），默认为0.0005。
    args = parser.parse_args()  # 这行代码通过调用 parse_args() 方法从命令行接收参数，解析并存储在 args 对象中。该对象将被用于访问所有由命令行提供的参数。

    # =================================================================================================
    # 覆盖默认参数值（如果提供了自定义参数字典）
    if custom_args:
        for key, value in custom_args.items():
            setattr(args, key, value)  # 动态设置参数
    # =================================================================================================

    args.steps_per_epoch = 10000000000
    args.data_dir = args.data_file+args.data_dir  # 构建数据目录路径:
    # os.environ['CUDA_VISIBLE_DEVICS'] = args.gpu_id  # 设置GPU环境变量: # 这里需要纠正错误
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id  # 假设 args.gpu_id = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id  # 假设 args.gpu_id = '0'

    os.makedirs(args.output, exist_ok=True)  # 创建输出目录:
    sys.stdout = Tee(os.path.join(args.output, 'out.txt'))  # 重定向标准输出和错误输出到文件:
    sys.stderr = Tee(os.path.join(args.output, 'err.txt'))
    args = act_param_init(args)  # 初始化特定活动参数:
    return args
