import torch
import torch.nn as nn
import torch.nn.functional as F

# DQN网络定义
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        # 增加网络的深度和宽度
        self.fc1 = nn.Linear(input_size, 128)  # 第一层
        self.fc2 = nn.Linear(128, 256)  # 增加的新层，更多的神经元
        self.fc3 = nn.Linear(256, 128)  # 又增加的新层，更多的神经元
        self.fc4 = nn.Linear(128, output_size)  # 输出层

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

