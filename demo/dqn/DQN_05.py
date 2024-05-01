import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import random

# 经验回放
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# DQN网络
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 环境
class TaskAllocationEnv:
    def __init__(self, tasks, nodes):
        self.tasks = np.array(tasks)
        self.nodes = np.array(nodes)
        self.num_tasks = len(tasks)
        self.num_nodes = len(nodes)
        self.action_space = len(tasks) * len(nodes)
        self.state_size = self.num_tasks + self.num_nodes

    def reset(self):
        self.task_status = np.zeros(self.num_tasks)  # 0 means unallocated
        self.node_status = self.nodes.copy()
        return self._get_state()

    def step(self, action):
        task_index = action // self.num_nodes
        node_index = action % self.num_nodes
        reward = 0
        done = False

        if self.task_status[task_index] == 0 and self.node_status[node_index] >= self.tasks[task_index]:
            self.node_status[node_index] -= self.tasks[task_index]
            self.task_status[task_index] = 1  # Mark as allocated
            reward = 1

        if all(self.task_status):
            done = True

        return self._get_state(), reward, done

    def _get_state(self):
        return np.concatenate((self.node_status, self.task_status), axis=0)

# Agent
class Agent:
    def __init__(self, state_size, action_size, epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=200):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(1000)
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.gamma = 0.8  # Discount factor
        self.batch_size = 128

    def select_action(self, state):
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                        math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if random.random() > eps_threshold:
            with torch.no_grad():
                return self.policy_net(torch.from_numpy(state).float().unsqueeze(0)).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_size)]], dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

# 训练代码略，因为这里主要展示结构和逻辑。

# 初始化环境和代理
tasks = [1, 9, 2, 7, 4, 3, 2, 5, 6, 1]
nodes = [10, 20, 30]
env = TaskAllocationEnv(tasks, nodes)
n_actions = len(tasks) * len(nodes)
n_states = env.state_size
agent = Agent(state_size=n_states, action_size=n_actions)

num_episodes = 100  # 训练的总回合数
for i_episode in range(num_episodes):
    # 初始化环境和状态
    state = env.reset()
    for t in range(100):  # 每回合的最大步骤数
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        # 选择并执行一个动作
        action = agent.select_action(state)
        next_state, reward, done = env.step(action.item())

        reward = torch.tensor([reward], dtype=torch.float)

        if not done:
            next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0)
        else:
            next_state_tensor = None

        # 存储转换
        agent.memory.push(state_tensor, action, next_state_tensor, reward, done)

        # 移动到下一个状态
        state = next_state

        # 执行优化步骤
        agent.optimize_model()
        if done:
            break
    # 更新目标网络
    if i_episode % 10 == 0:
        agent.target_net.load_state_dict(agent.policy_net.state_dict())

print("Training finished.")

