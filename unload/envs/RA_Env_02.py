import numpy as np

class ResourceAllocationEnv:
    def __init__(self, num_tasks, task_resources, num_nodes, node_resources):
        self.num_tasks = num_tasks
        self.task_resources = task_resources
        self.num_nodes = num_nodes
        self.node_resources = node_resources
        self.state = (tuple(self.node_resources), tuple(self.task_resources))

    def reset(self):
        self.node_resources = np.random.randint(1, 10, size=self.num_nodes)  # 重新随机生成节点资源
        self.state = (tuple(self.node_resources), tuple(self.task_resources))
        return self.state

    def step(self, action):
        node_index, task_index = action
        if self.node_resources[node_index] >= self.task_resources[task_index]:
            self.node_resources[node_index] -= self.task_resources[task_index]
            self.task_resources[task_index] = 0
        else:
            self.task_resources[task_index] -= self.node_resources[node_index]
            self.node_resources[node_index] = 0

        reward = self._calculate_reward()
        done = all(task == 0 for task in self.task_resources)
        self.state = (tuple(self.node_resources), tuple(self.task_resources))
        return self.state, reward, done, {}

    def _calculate_reward(self):
        # 可以根据自己的优化目标定义奖励函数，比如节点资源利用率的增加或者任务完成时间的减少
        return np.mean(self.node_resources) / np.sum(self.node_resources)

    def get_valid_actions(self):
        valid_actions = []
        for i in range(self.num_nodes):
            for j in range(self.num_tasks):
                if self.node_resources[i] >= self.task_resources[j]:
                    valid_actions.append((i, j))
        return valid_actions


