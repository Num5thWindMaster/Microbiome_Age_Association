# -*- coding: utf-8 -*-
# @Time    : 2024/9/20 15:34
# @Author  : HaiqingSun
# @OriginalFileName: data_processing
# @Software: PyCharm
# @AIMedicineWebPage: https://www.ilovemyhome.cc

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
import numpy as np

from utils import visualize_weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data = pd.read_csv('data3.csv')

# task = 'regression'
task = 'classification'
if task == 'regression':
    X = data.iloc[:, 3:].values  # 特征
    y = data.iloc[:, 1].values  # 标签
elif task == 'classification':
    X = data.iloc[:, 3:].values
    y = data.iloc[:, 2].values
    y = y
X, X_test, y, y_test = train_test_split(X, y, test_size=0.1, random_state=6)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)



class MLP(nn.Module):
    def __init__(self, input_size, dropout=0.1, task = 'regression'):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 1) if task == 'regression' else nn.Linear(32, 3)
        self.relu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)

        # Xavier 初始化
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # Dropout after first layer
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    def __init__(self, input_channels, num_classes, dropout=0.1):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * (input_size // 8), 128)  # Adjust input size based on conv/pooling layers
        self.fc2 = nn.Linear(128, num_classes if task == 'classification' else 1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension (for 1D convolution)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# 使用 CNN 模型代替 MLP
input_channels = 1  # 因为数据是一维的，可以设为 1
num_classes = 9  # 分类任务中的类别数


kf = KFold(n_splits=5, shuffle=True, random_state=6)
num_epochs = 80
batch_size = 32
dropout = 0.2
input_size = X.shape[1]

# # 初始化随机森林和决策树模型
# rf_model = RandomForestRegressor(n_estimators=100, random_state=6)
# dt_model = DecisionTreeRegressor(random_state=6)

fold = 1
total_loss = 0


for train_index, val_index in kf.split(X):
    print(f"Fold {fold}")

    # 获取当前折的训练集和验证集
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # 转换为张量
    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_torch = torch.tensor(X_val, dtype=torch.float32)
    y_val_torch = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    # 创建数据集和数据加载器
    train_dataset = TensorDataset(X_train_torch, y_train_torch)
    val_dataset = TensorDataset(X_val_torch, y_val_torch)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 初始化MLP模型
    model = MLP(input_size, dropout, task).to(device=device)
    # model = CNN(input_channels, num_classes, dropout).to(device)

    # 定义损失函数和优化器，定义任务类型
    if task == 'regression':
        criterion = nn.MSELoss()
    elif task == 'classification':
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    l1_lambda = 0.001
    # 训练MLP
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            if task == 'classification':
                targets = targets.view(-1).long()
            loss = criterion(outputs, targets)

            # L1 正则化
            l1_regularization = 0
            for param in model.parameters():
                l1_regularization += torch.norm(param, 1)

            loss += l1_lambda * l1_regularization
            # 反向传播和优化
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # 验证集上计算损失
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                val_outputs = model(inputs)
                if task == 'classification':
                    targets = targets.view(-1).long()
                val_loss += criterion(val_outputs, targets).item()

        val_loss /= len(val_loader)

        # if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # # 在验证集上用随机森林和决策树进行训练和预测
    # rf_model.fit(X, y)
    # dt_model.fit(X, y)

    # MLP模型的预测
    model.eval()

    with torch.no_grad():
        mlp_preds = model(X_test.to(device))
        # if task == 'regression':
        #     mlp_preds = mlp_preds.cpu().numpy()

    # # 随机森林和决策树的预测
    # rf_preds = rf_model.predict(X_test)
    # dt_preds = dt_model.predict(X_test)

    # 集成结果（平均）
    # ensemble_preds = (mlp_preds.flatten() + rf_preds + dt_preds) / 3

    # 计算均方误差
    # mse = mean_squared_error(y_test, ensemble_preds)
    fold_loss = criterion(mlp_preds.to(device), y_test.view(-1).long().to(device))
    total_loss += fold_loss
    print(f'Fold {fold} LOSS: {fold_loss:.4f}')
    fold += 1

test_loss = 0

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
torch.save(model.state_dict(), 'mlp_model.pth')
with torch.no_grad():
    correct = 0  # 用于统计分类正确的数量
    total = 0  # 用于统计测试集样本总数
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        test_outputs = model(inputs)

        # 处理分类任务
        if task == 'classification':
            targets = targets.view(-1).long()  # 转换目标为长整型以适应分类任务
            _, predicted = torch.max(test_outputs, 1)  # 获取预测结果中最大值的索引
            total += targets.size(0)  # 累计样本数量
            correct += (predicted == targets).sum().item()  # 统计预测正确的数量

        # 计算损失
        test_loss += criterion(test_outputs, targets).item()

    test_loss /= len(test_loader)

    # 打印损失
    print(f'Test Loss: {test_loss:.4f}')

    # 计算并打印分类准确率
    if task == 'classification':
        accuracy = correct / total * 100  # 计算准确率
        print(f'Classification Accuracy: {accuracy:.2f}%')

# 计算k折交叉验证的平均MSE
average_mse = total_loss / kf.get_n_splits()
print(f'Average loss across folds: {average_mse:.4f}')
# 可视化重要权重信息
visualize_weights(model)