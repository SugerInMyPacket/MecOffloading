import gym
from gym import spaces
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class TaskOffloadingEnv(gym.Env):
    def __init__(self, n_tasks, m_nodes, task_demands, node_resources):
        super(TaskOffloadingEnv, self).__init__()
        self.n_tasks = n_tasks
        self.m_nodes = m_nodes
        self.task_demands = np.array(task_demands)
        self.node_resources = np.array(node_resources)
        self.action_space = spaces.Discrete(m_nodes)  # 选择将任务卸载到哪个节点
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(m_nodes,), dtype=np.int)  # 节点资源状态
        self.current_task_index = 0

    def step(self, action):
        done = False
        reward = 0
        if self.node_resources[action] >= self.task_demands[self.current_task_index]:
            # 任务成功卸载
            self.node_resources[action] -= self.task_demands[self.current_task_index]
            reward = 1  # 基本奖励
            self.current_task_index += 1
            if self.current_task_index >= self.n_tasks:
                done = True
        else:
            # 资源不足，无法卸载
            reward = -1
            done = True  # 或者可以不结束让其尝试其他节点

        info = {}
        return self.node_resources.copy(), reward, done, info

    def reset(self):
        self.node_resources = np.array(self.node_resources)
        self.current_task_index = 0
        return self.node_resources.copy()

    def render(self, mode='human'):
        pass  # 可视化省略



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

def train_dqn(env):
    episodes = 500
    gamma = 0.99
    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 200
    target_update = 10
    batch_size = 5
    memory_size = 10000
    learning_rate = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_actions = env.action_space.n
    policy_net = DQN(env.observation_space.shape[0], n_actions).to(device)
    target_net = DQN(env.observation_space.shape[0], n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    # 训练循环省略...

    return policy_net


