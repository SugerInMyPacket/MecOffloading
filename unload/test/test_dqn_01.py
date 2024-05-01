import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import torch.nn.functional as F


from unload.env.AllocationEnv import AllocationEnv
from unload.env.DQN import DQN
from utils.DBUtils import select_data


def make_decisions(tasks, nodes, model_path):
    # 初始化环境
    env = AllocationEnv(tasks, nodes)
    #  加载模型
    model = load_model(model_path, env)
    # 用于存储每个任务的决策
    decisions = []

    state = env.reset()  # 重置环境状态
    state = torch.FloatTensor(state).unsqueeze(0)

    # 检查是否所有任务都已分配
    # while not env.done:
    #     with torch.no_grad():
    #         action = model(state).max(1)[1].view(1, 1).item()  # 使用模型做出决策
    #     state, _, done = env.step(action)  # 执行动作，获取下一状态
    #     state = torch.FloatTensor(state).unsqueeze(0)   # 准备模型输入
    #     decisions.append(action)
    #
    #     if done:
    #         break
    #
    #     return decisions

    while not env.done:  # 检查是否所有任务都已分配
        state = torch.FloatTensor(state).unsqueeze(0)  # 准备模型输入
        action = model(state).max(1)[1].view(1, 1).item()  # 使用模型做出决策
        next_state, _, done = env.step(action)  # 执行动作，获取下一状态
        decisions.append(action)  # 保存决策
        state = next_state  # 更新状态
        if done:
            break

    return decisions

def load_model(model_path, env):
    # 假设状态和动作空间由节点数确定
    model = DQN(len(env.nodes), len(env.nodes))
    # 加载模型
    model.load_state_dict(torch.load(model_path))
    model.eval()   # 设置为评估模式
    return model


def make_decisions2(tasks, nodes, model_path):
    """
    根据给定的任务和节点资源状态，使用训练好的模型来决定任务分配。

    :param tasks: 一个整数列表，表示每个任务需要的资源量。
    :param nodes: 一个整数列表，表示每个节点的初始资源量。
    :param model_path: 训练好的模型文件的路径。
    :return: 分配决策的数组，表示每个任务分配到的节点。
    """
    # 加载模型
    model = DQN(len(nodes), len(nodes))  # 假设状态和动作空间由节点数确定
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置为评估模式

    env = AllocationEnv(tasks, nodes)  # 初始化环境
    decisions = []  # 用于存储每个任务的决策
    state = env.reset()  # 重置环境状态

    while not env.done():  # 检查是否所有任务都已分配
        state = torch.FloatTensor(state).unsqueeze(0)  # 准备模型输入
        action = model(state).max(1)[1].view(1, 1).item()  # 使用模型做出决策
        next_state, _, done = env.step(action)  # 执行动作，获取下一状态
        # decisions.append(action)  # 保存决策
        state = next_state  # 更新状态

        if done:  # 如果所有任务都已尝试分配
            break

    return decisions


def make_decisions3(tasks, nodes, model_path):
    env = AllocationEnv(tasks, nodes)  # 假设环境已经正确设置
    # todo
    model = DQN(len(nodes), len(nodes) - 1)  # 确保这里的input_size和output_size与你的环境匹配
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()

    decisions = []
    state = env.reset()

    done = False
    # curr_step = 0
    # KeyNote: 修改策略，使得action多次无法实现done时，直接定为0
    while not done:
        # ++curr_step
        state = torch.FloatTensor(state).unsqueeze(0)  # 为模型准备当前状态
        with torch.no_grad():
            action = model(state).max(1)[1].view(1, 1).item()  # 选择最大Q值对应的动作

        # print(action)

        next_state, _, done = env.step(action, mode='test')
        # decisions.append(action)
        decisions.append(action)  # Todo
        state = next_state

    return decisions


def preprocess_input(input_array, fixed_length, padding_value=0):
    """
    将输入数组预处理为固定长度。
    :param input_array: 原始输入数组。
    :param fixed_length: 目标固定长度。
    :param padding_value: 用于填充的值。
    :return: 预处理后的固定长度数组。
    """
    if len(input_array) < fixed_length:
        # 如果数组长度小于固定长度，进行填充
        padded_array = np.pad(input_array, (0, fixed_length - len(input_array)), 'constant', constant_values=padding_value)
        return padded_array
    elif len(input_array) > fixed_length:
        # 如果数组长度大于固定长度，进行截断
        return input_array[:fixed_length]
    else:
        # 长度相等，直接返回
        return input_array


if __name__ == '__main__':
    # 使用训练好的模型做出决策
    # 假设有新的任务和资源数组
    # task_fixed_length = 10
    # new_tasks = [3, 8, 2, 4, 5]
    # # new_tasks = preprocess_input(new_tasks, task_fixed_length, 0)
    # node_fixed_length = 4
    # # new_nodes = [30, 10, 15, 20]
    #
    sql_database_name = "task4"
    task_demands = []
    results = select_data(sql_database_name)

    index = 0
    for r in results:
        task_demands.append(r[1] * r[3])
        if random.random() < 0.1:
            # 随机使得一些需求为 0
            task_demands[index] = 0

        ++index
    # task_demands = [0, 0, 0, 0, 0, 0]
    print(task_demands)
    new_nodes = [20000, 2500, 800, 590, 620, 550, 450, 780, 990, 680, 780, 980, 0]
    # new_nodes = preprocess_input(new_nodes, node_fixed_length, 0)
    model_path = '../mnt/unload_dqn_v10_task100.pth'
    decisions = make_decisions3(task_demands, new_nodes, model_path)
    print(decisions)

    for i in range(len(task_demands)):
        new_nodes[decisions[i]] -= task_demands[i]
    print(new_nodes)

