import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import random
import os

# 设定随机种子以确保可复现性
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)

# 经验回放
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

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
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.network(x)

# 环境
class TaskAllocationEnv:
    def __init__(self, tasks, nodes):
        self.tasks = np.array(tasks)
        self.nodes = np.array(nodes)
        self.num_tasks = len(tasks)
        self.num_nodes = len(nodes)

    def reset(self):
        self.task_status = np.zeros(self.num_tasks)
        self.node_status = self.nodes.copy()
        return np.concatenate((self.node_status, self.task_status))

    def step(self, action):
        task_index = action // self.num_nodes
        node_index = action % self.num_nodes
        reward = 0
        done = False

        if self.task_status[task_index] == 0 and self.node_status[node_index] >= self.tasks[task_index]:
            self.node_status[node_index] -= self.tasks[task_index]
            self.task_status[task_index] = 1
            reward = 1

        if all(self.task_status):
            done = True

        return np.concatenate((self.node_status, self.task_status)), reward, done

# 代理
class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(10000)
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.steps_done = 0
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 200
        self.gamma = 0.8  # 折扣因子
        self.batch_size = 128

    def select_action(self, state):
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                        np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if random.random() > eps_threshold:
            with torch.no_grad():
                return self.policy_net(torch.tensor(state, dtype=torch.float).unsqueeze(0)).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_size)]], dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
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
        self.optimizer.step()

# 初始化环境和代理
tasks = [1, 9, 2, 7, 4, 3, 2, 5, 6, 1]
nodes = [10, 20, 30]
env = TaskAllocationEnv(tasks, nodes)
agent = Agent(state_size=len(tasks) + len(nodes), action_size=len(tasks) * len(nodes))

# 训练循环
num_episodes = 100
for i_episode in range(num_episodes):
    state = env.reset()
    for t in range(100):
        action = agent.select_action(state).item()
        next_state, reward, done = env.step(action)
        if done:
            next_state = None

        agent.memory.push(torch.tensor([state], dtype=torch.float),
                          torch.tensor([[action]], dtype=torch.long),
                          None if next_state is None else torch.tensor([next_state], dtype=torch.float),
                          torch.tensor([reward], dtype=torch.float),
                          done)

        state = next_state
        agent.optimize_model()
        if done:
            break

    if i_episode % 10 == 0:
        agent.target_net.load_state_dict(agent.policy_net.state_dict())

# 保存模型
model_path = '/mnt/data/task_allocation_model.pth'
torch.save(agent.policy_net.state_dict(), model_path)
print(f'Model saved to {model_path}')
