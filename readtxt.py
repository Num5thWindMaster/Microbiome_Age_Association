# -*- coding: utf-8 -*-
# @Time    : 2024/9/21 21:26
# @Author  : HaiqingSun
# @OriginalFileName: readtxt
# @Software: PyCharm
# @AIMedicineWebPage: https://www.ilovemyhome.cc

with open("fc1.weight_weights.txt", "r") as f:
    file = f.readlines()
    print(len(file))
    list1 = file[1].split()
    print(len(list1))

import numpy as np

# 1. 读取权重文件，假设文件名为'weights.txt'
weights = np.loadtxt('fc1.weight_weights.txt')

# 2. 计算每个特征在所有神经元上的重要性
# 我们对每一列（每个特征）的权重取绝对值，并在所有输出神经元上累加
feature_importance = np.sum(np.abs(weights), axis=0)  # 对列取绝对值并累加

# 3. 找到最重要的20个权重对应的特征索引
top_20_indices = np.argsort(feature_importance)[-20:][::-1]  # 从大到小排序，取前20个

# 4. 输出最重要的特征索引
print("Top 20 most important feature indices:")
print(top_20_indices)

import pandas as pd

# 1. 读取 CSV 文件
data = pd.read_csv('data.csv')

# 2. 获取列名列表
column_names = data.columns

# 3. 根据索引获取对应的列名
important_column_names = column_names[top_20_indices+3]
important_weights = feature_importance[top_20_indices]

# 4. 将列名和对应权重配对并保存到txt文件中，每行列名和权重用空格隔开
with open('important_columns_with_weights3.txt', 'w') as f:
    for col_name, weight in zip(important_column_names, important_weights):
        f.write(f'{col_name} {weight:.3f}\n')

print("Important columns and weights saved to important_columns_with_weights.txt")