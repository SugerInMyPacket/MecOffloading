import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import torch.nn.functional as F


# from AllocationEnv import AllocationEnv
from unload.env.AllocationEnv import AllocationEnv
from unload.env.DQN import DQN

from utils.DBUtils import select_data


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


def select_action_softmax(state, policy_net):
    """
    使用softmax选择动作的函数。
    :param state: 当前环境状态的Tensor。
    :param policy_net: 策略网络，用于计算每个动作的Q值。
    :return: 选择的动作。
    """
    # 使用策略网络计算当前状态下每个动作的Q值
    with torch.no_grad():
        q_values = policy_net(state)
    # 计算softmax概率分布
    probabilities = F.softmax(q_values, dim=1)
    # 根据概率分布选择动作
    action = probabilities.multinomial(num_samples=1)
    if action < 0:
        action = 0
    if action >= len(state):
        action = len(state) - 1
    return action


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


# 训练循环加入ε-greedy策略和目标网络
def train_dqn(env,
              episodes=1000, batch_size=32, gamma=0.99,
              epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=200,
              TARGET_UPDATE = 200, max_steps_per_episode=100):
    # train_dqn 负责管理训练周期、与环境交互、更新模型参数以及评估模型性能

    # 定义动作空间的大小，即可选择的节点数量
    # n_actions = len(env.nodes)
    n_actions = len(env.nodes) - 1   # todo

    # 初始化策略网络
    policy_net = DQN(len(env.nodes), n_actions)  # todo

    # 初始化目标网络，并复制策略网络的权重
    target_net = DQN(len(env.nodes), n_actions)  # todo

    target_net.load_state_dict(policy_net.state_dict())
    # 设置目标网络为评估模式，这是因为我们只在目标网络上进行前向传播
    target_net.eval()
    # 选择优化器
    optimizer = optim.Adam(policy_net.parameters())
    # 初始化经验回放缓存
    memory = ReplayMemory(10000)

    # 用于epsilon的递减计算
    steps_done = 0
    for episode in range(episodes):
        total_reward = 0
        # 重置环境并初始化状态
        state = env.reset()
        # 转换为合适的张量格式
        state = torch.FloatTensor(state).unsqueeze(0)
        done = False

        # 当前 episode 训练到一定次数后跳出
        # todo
        # for step_one_episode in range(max_steps_per_episode):
        while True:

            # 根据epsilon贪心策略选择动作
            # todo
            # epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
            #           math.exp(-1. * steps_done / epsilon_decay)
            epsilon = 0.5
            steps_done += 1
            action = select_action(state, policy_net, epsilon, n_actions)
            # 在环境中执行动作，观察下一个状态和奖励
            next_state, reward, done = env.step(action.item())

            # 使用softmax策略选择动作
            # action = select_action_softmax(state, policy_net)
            # next_state, reward, done = env.step(action.item())

            # todo
            # if done or step_one_episode == max_steps_per_episode:
                # 如果任务完成或达到最大步骤数，结束回合
            if done:
                break

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
    torch.save(policy_net.state_dict(), '../mnt/unload_dqn_v10_task100_per100.pth')


if __name__ == '__main__':
    # 实例化环境并开始训练，包括环境设置和参数调整
    # tasks = [5, 10, 3, 6]  # 任务资源需求示例
    # nodes = [10, 10, 10]  # 节点资源示例
    # env = AllocationEnv(tasks, nodes)
    sql_database_name = "task4"

    task_demands = []
    results = select_data(sql_database_name)

    index = 0
    for r in results:
        task_demands.append(r[1] * r[3])
        # if random.random() < 0.1:
        #     task_demands[index] = 0
        ++index

    node_resources = [20000, 2500, 800, 590, 620, 550, 450, 780, 990, 680, 780, 980, 0]
    env = AllocationEnv(task_demands, node_resources)

    train_dqn(env)


