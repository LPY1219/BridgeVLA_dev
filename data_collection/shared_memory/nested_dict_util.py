"""
嵌套字典工具函数

提供对嵌套字典结构的映射和检查操作。
"""

import functools


def nested_dict_map(f, x):
    """
    对嵌套字典x的所有叶子节点应用函数f
    
    Args:
        f: 要应用的函数
        x: 嵌套字典或值
        
    Returns:
        应用函数后的结果
    """
    if not isinstance(x, dict):
        return f(x)
    y = dict()
    for key, value in x.items():
        y[key] = nested_dict_map(f, value)
    return y


def nested_dict_reduce(f, x):
    """
    对嵌套字典x的所有值应用函数f，并归约为单一值
    
    Args:
        f: 归约函数
        x: 嵌套字典或值
        
    Returns:
        归约后的单一值
    """
    if not isinstance(x, dict):
        return x

    reduced_values = list()
    for value in x.values():
        reduced_values.append(nested_dict_reduce(f, value))
    y = functools.reduce(f, reduced_values)
    return y


def nested_dict_check(f, x):
    """
    检查嵌套字典的所有叶子节点是否满足条件f
    
    Args:
        f: 检查函数，返回布尔值
        x: 嵌套字典或值
        
    Returns:
        所有叶子节点都满足条件时返回True，否则返回False
    """
    bool_dict = nested_dict_map(f, x)
    result = nested_dict_reduce(lambda x, y: x and y, bool_dict)
    return result
