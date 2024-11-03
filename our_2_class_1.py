from train import main, get_args  # 导入主函数和参数函数
from itertools import product
from tqdm import tqdm  # 导入tqdm库


def run_experiment(modified_params):
    # 从 get_args() 获取默认参数
    args = get_args(modified_params)

    # # 修改感兴趣的参数
    # for key, value in modified_params.items():
    #     setattr(args, key, value)

    # 调用训练函数
    main(args)


# 定义需要变化的参数网格
param_grid = {
    'latent_domain_num': [i for i in range(2, 23)],  # 不同的 latent domain 数量
    'lr': [1e-2],  # 不同的学习率
    'test_envs': [[0], [1], [2], [3]],  # 不同的测试环境
    'activity_id': [i for i in range(1, 2)]
}

# 使用 itertools.product 生成参数的所有组合
keys, values = zip(*param_grid.items())
param_combinations = [dict(zip(keys, v)) for v in product(*values)]

# 循环遍历所有参数组合并运行实验
# 使用 tqdm 包裹 param_combinations，显示进度条
for param_set in tqdm(param_combinations, desc="Running experiments"):
    run_experiment(param_set)
