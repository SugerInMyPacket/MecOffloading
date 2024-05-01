import numpy as np


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
        self.current_node_states = np.concatenate((nodes, tasks))  # 节点当前状态（剩余资源）
        self.task_index = 0  # 当前处理的任务索引

        # 为 test 而加
        # self.done = False
        self.train_one_task_action_times = 0


    def reset(self):
        """
        重置环境到初始状态，用于开始一个新的回合
        :return: 返回环境的初始状态
        """
        self.current_node_states = np.concatenate((self.nodes, self.tasks)) # 重置节点状态为初始资源量
        self.task_index = 0 # 重置当前任务索引
        # 返回环境的初始状态

        # todo
        # self.current_node_states.append(self.tasks[self.task_index])
        # self.current_node_states[-1] = self.tasks[self.task_index]
        return self.current_node_states

    def step(self, action, mode='train'):
        """
        在环境中执行一个动作
        :param action: 一个整数，表示选择的节点索引
        :return: next_state, reward, done，分别代表动作执行后的新状态、奖励和是否结束
        """

        # todo
        num_tasks = len(self.tasks)
        node_index = action // num_tasks
        task_index = action % num_tasks

        # task_demand = self.tasks[self.task_index]
        task_demand = self.tasks[task_index]  # todo
        reward = 0
        done = False   # 初始化done标志为 False

        # todo
        flag_next = False

        # 如果选定的节点有足够的资源执行当前任务
        # if self.current_node_states[action] >= task_demand:
        if self.current_node_states[node_index] >= task_demand: # todo

            # self.current_node_states[action] -= task_demand   # 执行任务，减少节点资源
            self.current_node_states[node_index] -= task_demand   # todo

            reward = 60 # 成功分配奖励
            self.task_index += 1 # 移动到下一个任务

            # todo
            flag_next = True

            # 选择不同的节点，可以给不同的奖励
            if node_index == 0:
                reward = -20

        else:
            if mode == 'test':
                node_index = 0
                self.current_node_states[node_index] -= task_demand  # 执行任务，减少节点资源
                self.task_index += 1  # 移动到下一个任务

            if mode == 'train':
                self.train_one_task_action_times += 1
            # 如果选定的节点没有足够的资源
            reward = -40   # 分配失败惩罚

            if self.train_one_task_action_times > 100:
                self.train_one_task_action_times = 0
                node_index = 0
                self.current_node_states[node_index] -= task_demand  # 执行任务，减少节点资源
                self.task_index += 1  # 移动到下一个任务

        # 检查是否所有任务都已尝试分配，以确定是否结束
        if self.task_index >= len(self.tasks):
            done = True  # 所有任务都已尝试分配，设置done为True

        # todo
        else:
            if flag_next:
                self.current_node_states[-1] = self.tasks[self.task_index]

        return self.current_node_states, reward, done
