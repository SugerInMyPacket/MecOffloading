
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class AllocationEnv:
    def __init__(self, nodes):
        self.nodes = np.array(nodes)  # 节点的初始资源量
        self.state = None

    def reset(self, tasks):
        self.state = np.array([self.nodes, tasks])
        return self.state

    def step(self, action):
        # 简化：action是选择的节点索引，state[1][0]是当前任务的需求
        task_demand = self.state[1][0]
        if self.state[0][action] >= task_demand:
            self.state[0][action] -= task_demand  # 更新节点资源
            reward = 1
        else:
            reward = -1

        self.state = np.delete(self.state, 1, 1)  # 移除已分配的任务
        done = len(self.state[1]) == 0  # 检查是否所有任务都已处理
        return self.state, reward, done

    def get_state(self):
        # 返回当前状态
        return self.state


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x, hidden = self.gru(x, hidden)
        x = self.fc(x[:, -1, :])  # 我们只关心序列的最后输出
        return x, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)


def train_model(model, env, tasks, epochs=100, learning_rate=0.01):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        hidden = model.init_hidden(1)
        state = env.reset(tasks)
        cum_loss = 0

        for task_demand in tasks:
            # 假设状态是当前节点资源和任务需求的简单拼接
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_pred, hidden = model(state_tensor, hidden)

            # 为简化，假设每次只处理一个任务，动作是选择节点
            action_true = torch.tensor([np.random.randint(len(nodes))])  # 随机选择节点作为示例
            loss = loss_fn(action_pred, action_true)
            cum_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 环境步进
            state, _, done = env.step(action_true.item())
            if done:
                break

        print(f"Epoch {epoch+1}, Loss: {cum_loss/len(tasks)}")

def make_decisions(model, env, tasks):
    hidden = model.init_hidden(1)
    state = env.reset(tasks)
    decisions = []

    for task_demand in tasks:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_pred, hidden = model(state_tensor, hidden)
            action = action_pred.argmax(1).item()
        decisions.append(action)
        state, _, done = env.step(action)
        if done:
            break

    return decisions
