# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from torchvision import transforms
import numpy as np


def act_train():  # 这段代码定义了一个函数 act_train()，它返回一个由 transforms.Compose 组合的图像处理流水线，包含了一个将数据转换为 PyTorch 张量的步骤。具体解释如下：
    return transforms.Compose([  #
        transforms.ToTensor()
    ])
# 这是 PyTorch 中的一个函数，用于将多个图像转换操作组合成一个流水线。在这里，它将单个操作 transforms.ToTensor() 组合起来。
# 作用：将多个图像处理步骤按顺序组合，使得每次调用时都可以一次性应用这些转换。
def loaddata_from_numpy(dataset='dsads', task='cross_people', root_dir='./data/act/', args=None):
    if dataset == 'pamap' and task == 'cross_people':
        x = np.load(root_dir+dataset+'/'+dataset+'_x1.npy')
        ty = np.load(root_dir+dataset+'/'+dataset+'_y1.npy')
    elif dataset == 'pads':
        # x = np.load(root_dir+dataset+'/'+dataset+f'_{args.activity_id}_x.npy')  # 加载的数据：x 是输入数据的数组，可能包含来自传感器或其他来源的特征数据。
        # ty = np.load(root_dir+dataset+'/'+dataset+f'_{args.activity_id}_y.npy')
        x = np.load(root_dir+dataset+'/'+dataset+f'_{args.activity_id}_x.npy')  # 加载的数据：x 是输入数据的数组，可能包含来自传感器或其他来源的特征数据。
        ty = np.load(root_dir+dataset+'/'+dataset+f'_{args.activity_id}_y.npy')
    else:
        x = np.load(root_dir+dataset+'/'+dataset+'_x.npy')  # 加载的数据：x 是输入数据的数组，可能包含来自传感器或其他来源的特征数据。
        ty = np.load(root_dir+dataset+'/'+dataset+'_y.npy')  # ty 是标签数据的数组，其中每行可能包含多个标签信息，如类别、参与者、传感器位置等。
    cy, py, sy = ty[:, 0], ty[:, 1], ty[:, 2]  # 作为类别标签 cy。每个元素代表该样本的类别（如动作类型）。# 从标签数组中提取第 1 列，作为参与者标签 py。每个元素代表数据所属的参与者。 # ty[:, 2]：从标签数组中提取第 2 列，作为传感器位置标签 sy。每个元素代表传感器数据的位置或类型。
    return x, cy, py, sy

# loaddata_from_numpy 函数用于从指定路径加载数据集中的 numpy 文件，并返回数据和相应的标签。
# 这是一个数据加载的基础函数，特别用于跨参与者任务中的数据处理。下面是对每一行代码的详细解释：
# 定义函数：loaddata_from_numpy 函数用于从指定的路径加载 numpy 格式的数据。
# 参数：
# dataset='dsads'：指定数据集名称，默认是 'dsads' 数据集。
# task='cross_people'：指定任务类型，默认是 'cross_people'，表示跨参与者任务。
# root_dir='./data/act/'：指定数据集所在的根目录，默认路径是 './data/act/'。
# 条件加载数据
# 条件判断：检查是否数据集为 'pamap' 且任务为 'cross_people'。
# 加载数据：
# x = np.load(root_dir + dataset + '/' + dataset + '_x1.npy')：
# 使用 np.load 从指定路径加载输入数据（特征），路径由根目录、数据集名称和文件名组合而成。
# x：加载的数据，是一个 numpy 数组，通常包含模型输入的特征数据。
# ty = np.load(root_dir + dataset + '/' + dataset + '_y1.npy')：
# 从指定路径加载标签数据，标签包含类别、参与者和传感器位置等信息。
# ty：加载的标签数据，通常是一个二维数组，包含多个类型的标签信息。
# 默认加载数据
# 默认加载：对于其他数据集和任务组合，执行默认的数据加载操作。
# 加载数据：
# x = np.load(root_dir + dataset + '/' + dataset + '_x.npy')：
# 加载输入数据，路径与上面的类似，只是文件名不同（没有1 后缀）。
# ty = np.load(root_dir + dataset + '/' + dataset + '_y.npy')：
# 加载标签数据，文件名不同（没有 1 后缀）。
# 分割标签
# 拆分标签：将标签数据 ty 分成三部分：
# cy = ty[:, 0]：提取类别标签 cy，对应标签数据的第一列，通常是任务的主要目标（如动作类型）。
# py = ty[:, 1]：提取参与者标签 py，对应标签数据的第二列，用于标识数据的来源（如不同的参与者）。
# sy = ty[:, 2]：提取传感器位置标签 sy，对应标签数据的第三列，用于标识传感器的位置或类型。
# 返回值：
# x：返回加载的输入数据，形状通常为 (样本数, 特征数)。
# cy：返回类别标签数组。
# py：返回参与者标签数组。
# sy：返回传感器位置标签数组。
# 函数的目的
# 数据加载：根据数据集名称和任务类型，动态选择加载不同的 numpy 文件，提供灵活的数据加载方式。
# 标签拆分：将标签数据按类别、参与者和位置拆分，支持不同粒度的标签管理，这在跨域任务中非常重要。
# 结合论文逻辑的总结
# 跨参与者任务：通过加载不同参与者的数据并拆分标签，函数支持跨域学习的设置，符合论文中处理多样化和多源数据的需求。
# 灵活性：根据条件选择不同的文件，表明设计时考虑到了不同任务的特殊需求，提高了代码的通用性和灵活性。
# 标签管理：按类别、参与者和传感器位置拆分标签，支持模型在多标签环境下的训练，有助于提升模型的泛化能力。

