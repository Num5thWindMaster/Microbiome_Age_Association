# -*- coding: utf-8 -*-
# @Time    : 2024/9/21 13:20
# @Author  : HaiqingSun
# @OriginalFileName: utils
# @Software: PyCharm
# @AIMedicineWebPage: https://www.ilovemyhome.cc
import numpy as np
from matplotlib import pyplot as plt


def visualize_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights = param.data.cpu().numpy()
            plt.figure(figsize=(10, 6))
            plt.hist(weights.flatten(), bins=50, color='b', alpha=0.7)
            plt.title(f'Weight Distribution of {name}')
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')
            plt.show()
            np.savetxt(name + "_weights.txt", weights, fmt='%f')