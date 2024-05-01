import numpy as np


class AllocationEnv:
    def __init__(self, nodes, tasks):
        self.nodes = np.array(nodes)
        self.tasks = np.array(tasks)
        self.current_state = None
        self.done = False

    def reset(self):
        self.current_state = np.concatenate((self.nodes, self.tasks), axis=None)
        self.done = False
        return self.current_state

    def step(self, action):
        task_demand = self.current_state[len(self.nodes)]
        if self.current_state[action] >= task_demand:
            self.current_state[action] -= task_demand
            reward = 1
        else:
            reward = -1  # Penalty for trying to allocate to an insufficient node

        self.current_state = np.delete(self.current_state, len(self.nodes))  # Remove the allocated task
        if len(self.current_state) == len(self.nodes):  # All tasks are processed
            self.done = True

        return self.current_state, reward, self.done



import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        x, hidden = self.gru(x, hidden)
        x = self.fc(x[:, -1, :])  # 只关心序列的最后输出
        return x

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)


import torch.optim as optim
import torch.nn.functional as F


# 训练模型
def train_model(model, env, tasks, epochs=200, learning_rate=0.01):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        state = env.reset()
        state_tensor = torch.FloatTensor(state).view(1, 1, -1)
        cum_loss = 0

        for task in tasks:
            action_scores = model(state_tensor)
            action = torch.argmax(action_scores, dim=1)

            next_state, reward, done = env.step(action.item())
            next_state_tensor = torch.FloatTensor(next_state).view(1, 1, -1)

            # 使用MSE作为简化示例；实际应用中可能需要更合适的损失函数
            target = torch.FloatTensor([reward])  # 简化目标；实际中可能复杂
            loss = F.mse_loss(action_scores.squeeze(), target)
            cum_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state_tensor = next_state_tensor

            if done:
                break

        print(f"Epoch {epoch + 1}, Loss: {cum_loss / len(tasks)}")


def make_decisions(model, env, tasks):
    state = env.reset(tasks)
    state_tensor = torch.FloatTensor(state).view(1, 1, -1)
    decisions = []

    for _ in range(len(tasks)):
        with torch.no_grad():
            action_scores = model(state_tensor)
            action = torch.argmax(action_scores, dim=1).item()
        decisions.append(action)

        next_state, _, done = env.step(action)
        next_state_tensor = torch.FloatTensor(next_state).view(1, 1, -1)

        state_tensor = next_state_tensor

        if done:
            break

    return decisions


if __name__ == '__main__':
    # 初始化环境
    nodes = [10, 15, 20]  # 节点资源
    tasks = [5, 10, 2, 8]  # 任务的资源需求
    env = AllocationEnv(nodes, tasks)

    # 初始化RNN模型
    input_size = len(nodes) + 1  # 假设状态是节点资源加一个任务需求
    hidden_size = 128
    output_size = len(nodes)  # 输出是每个节点的动作得分
    # model = RNN(input_size, hidden_size, output_size)
    # 模型的输入大小是节点数加任务数，输出大小是节点数
    model = RNN(input_size=len(nodes) + len(tasks), hidden_size=16, output_size=len(nodes))
    train_model(model, env, tasks, epochs=200, learning_rate=0.005)



def test():
    # 测试模型
    new_tasks = [3, 7, 4, 6]  # 新任务序列
    decisions = make_decisions(model, env, new_tasks)
    print("Decisions:", decisions)
