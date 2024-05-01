import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


# 环境定义
class AllocationEnv:
    """
    初始化环境
        :param tasks: 一个整数列表，表示每个任务需要的资源量
        :param nodes: 一个整数列表，表示每个节点的初始资源量
    """
    def __init__(self, tasks, nodes):
        self.tasks = tasks  # 任务资源需求列表
        self.nodes = nodes  # 节点初始资源列表
        self.current_node_states = np.array(nodes)  # 节点当前状态（剩余资源）
        self.task_index = 0  # 当前处理的任务索引


    def reset(self):
        """
        重置环境到初始状态，用于开始一个新的回合
        :return: 返回环境的初始状态
        """
        self.current_node_states = np.array(self.nodes) # 重置节点状态为初始资源量
        self.task_index = 0 # 重置当前任务索引
        # 返回环境的初始状态
        return self.current_node_states

    def step(self, action):
        """
        在环境中执行一个动作
        :param action: 一个整数，表示选择的节点索引
        :return: next_state, reward, done，分别代表动作执行后的新状态、奖励和是否结束
        """

        task_demand = self.tasks[self.task_index]
        reward = 0
        done = False # 初始化done标志为False

        # 如果选定的节点有足够的资源执行当前任务
        if self.current_node_states[action] >= task_demand:
            self.current_node_states[action] -= task_demand   # 执行任务，减少节点资源
            reward = 1  # 成功分配奖励
            self.task_index += 1 # 移动到下一个任务
        else:
            # 如果选定的节点没有足够的资源
            reward = -3  # 分配失败惩罚

        # 检查是否所有任务都已尝试分配，以确定是否结束
        if self.task_index >= len(self.tasks):
            done = True  # 所有任务都已尝试分配，设置done为True

        return self.current_node_states, reward, done


# ε-greedy策略
def select_action(state, policy_net, epsilon, n_actions):
    # 根据ε-greedy策略随机选择动作或根据模型预测选择动作
    if random.random() > epsilon:
        # 不随机选择，而是让模型根据当前状态预测动作
        with torch.no_grad():
            # model(state)会返回每个动作的预期收益（即Q值）
            # max(1)[1]选出这些收益中最大的一个，即选择最优动作
            # view(1, 1)确保输出的格式符合预期
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        # 以ε的概率随机选择一个动作
        # 这里使用torch.tensor()来创建一个包含随机动作的tensor
        # 这是因为后续的操作都是基于tensor的
        return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)

# DQN网络定义
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.fc(x)


# 经验回放
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, next_state, reward):
        self.memory.append((state, action, next_state, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# 训练过程（完整）
# 目的是通过从经验回放缓存中采样一批经历（transitions）来更新策略网络（policy network），从而使模型学习最优策略
def optimize_model(memory, policy_net, target_net, optimizer, batch_size, gamma):
    # 1. 检查经验回放中是否有足够的样本进行训练
    if len(memory) < batch_size:
        return
    # 2. 从经验回放中随机采样一批样本
    transitions = memory.sample(batch_size)
    batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

    # 3. 将采样的数据转换为PyTorch张量
    batch_state = torch.cat(batch_state)
    batch_action = torch.cat(batch_action)
    batch_reward = torch.cat(batch_reward)
    batch_next_state = torch.cat(batch_next_state)

    # 4. 计算当前状态-动作对的Q值。这些是我们模型预测的当前状态下采取特定动作的值。
    # gather(1, batch_action) 确保我们只选择了执行的动作对应的Q值。
    state_action_values = policy_net(batch_state).gather(1, batch_action)

    # 计算所有下一个状态的最大预期Q值。
    # detach() 用于防止这些值被认为是当前模型参数的梯度的一部分，
    # 因为目标值应当是固定的，对于当前模型参数的更新不应该有影响。
    next_state_values = target_net(batch_next_state).max(1)[0].detach()
    # 计算下一个状态的Q值的期望值，包括奖励和下一个状态的最大Q值。
    # 这是根据贝尔曼方程计算的。
    expected_state_action_values = (next_state_values * gamma) + batch_reward

    # 计算损失
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # 优化模型
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


TARGET_UPDATE = 100
# 训练循环加入ε-greedy策略和目标网络
def train_dqn(env, episodes=500, batch_size=32, gamma=0.99,
              epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=200):
    # train_dqn 负责管理训练周期、与环境交互、更新模型参数以及评估模型性能

    # 定义动作空间的大小，即可选择的节点数量
    n_actions = len(env.nodes)
    # 初始化策略网络
    policy_net = DQN(len(env.nodes), n_actions)
    # 初始化目标网络，并复制策略网络的权重
    target_net = DQN(len(env.nodes), n_actions)
    target_net.load_state_dict(policy_net.state_dict())
    # 设置目标网络为评估模式，这是因为我们只在目标网络上进行前向传播
    target_net.eval()
    # 选择优化器
    optimizer = optim.Adam(policy_net.parameters())
    # 初始化经验回放缓存
    memory = ReplayMemory(10000)

    # 用于epsilon的递减计算
    steps_done = 0
    total_reward = 0
    for episode in range(episodes):
        # 重置环境并初始化状态
        state = env.reset()
        # 转换为合适的张量格式
        state = torch.FloatTensor(state).unsqueeze(0)
        done = False

        while not done:
            # 根据epsilon贪心策略选择动作
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                      math.exp(-1. * steps_done / epsilon_decay)
            steps_done += 1
            action = select_action(state, policy_net, epsilon, n_actions)
            # 在环境中执行动作，观察下一个状态和奖励
            next_state, reward, done = env.step(action.item())
            reward = torch.tensor([reward], dtype=torch.float)

            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            # 将状态转换、动作、奖励和下一个状态的转换存储在经验回放中
            memory.push(state, action, next_state, reward)
            # 移动到下一个状态
            state = next_state

            # 执行一步优化（在策略网络上）
            optimize_model(memory, policy_net, target_net, optimizer, batch_size, gamma)

            total_reward += reward
            print(f"Episode {episode + 1}/{episodes} completed, Total Reward: {total_reward}")

        # 更新目标网络的参数
        # 定期更新目标网络的权重
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    # 训练结束后保存模型
    torch.save(policy_net.state_dict(), '../mnt/data/my_dqn_model.pth')


# 实例化环境并开始训练，包括环境设置和参数调整
tasks = [5, 10, 3, 6]  # 任务资源需求示例
nodes = [10, 10, 10]  # 节点资源示例
env = AllocationEnv(tasks, nodes)
train_dqn(env)


def make_decisions(tasks, nodes, model_path='../mnt/data/my_dqn_model.pth'):
    env = AllocationEnv(tasks, nodes)
    model = load_model(model_path, env)
    decisions = []

    state = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0)

    while not env.done():
        with torch.no_grad():
            action = model(state).max(1)[1].view(1, 1).item()
        state, _, done = env.step(action)
        state = torch.FloatTensor(state).unsqueeze(0)
        decisions.append(action)

        if done:
            break

    return decisions

def load_model(model_path, env):
    model = DQN(len(env.nodes), len(env.nodes))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def test():
    # 使用训练好的模型做出决策
    # 假设有新的任务和资源数组
    new_tasks = [3, 8, 2, 4, 5]
    new_nodes = [15, 10, 20]
    decisions = make_decisions(new_tasks, new_nodes)
    print(decisions)

