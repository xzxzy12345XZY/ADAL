# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from .algs.diversify import Diversify

ALGORITHMS = [
    'diversify'
]


def get_algorithm_class(algorithm_name):  # 定义函数：get_algorithm_class 是一个用于返回指定算法类的函数。
    if algorithm_name not in ALGORITHMS:  # 作用：检查给定的算法名称 algorithm_name 是否存在于预定义的算法集合 ALGORITHMS 中.
        raise NotImplementedError(
            "Algorithm not found: {}".format(algorithm_name))
    return Diversify
