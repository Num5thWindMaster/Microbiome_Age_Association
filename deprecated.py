# -*- coding: utf-8 -*-
# @Time    : 2024/9/21 13:23
# @Author  : HaiqingSun
# @OriginalFileName: deprecated
# @Software: PyCharm
# @AIMedicineWebPage: https://www.ilovemyhome.cc

# # 将数据集分为训练集和测试集
# X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.2, random_state=6)
#
# # 将剩下的20%再分成10%验证集和10%测试集
# X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=6)
#
# # 将 NumPy 数组转换为 PyTorch 张量
# X_train_torch = torch.tensor(X_train, dtype=torch.float32)
# y_train_torch = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
# X_test_torch = torch.tensor(X_test, dtype=torch.float32)
# y_test_torch = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
# X_val_torch = torch.tensor(X_val, dtype=torch.float32)
# y_val_torch = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
#
# # 创建数据集和数据加载器
# batch_size = 32
#
# train_dataset = TensorDataset(X_train_torch, y_train_torch)
# val_dataset = TensorDataset(X_val_torch, y_val_torch)
#
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size)
#