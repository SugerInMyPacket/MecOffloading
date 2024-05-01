import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import count

# 环境定义
class AllocationEnv:
    def __init__(self, tasks, nodes):
        self.tasks = tasks  # 任务所需资源列表
        self.nodes = nodes  # 节点初始资源列表
        # self.state = None
        self.state = np.array(self.nodes + self.tasks)
        self.done = False

    def reset(self):
        self.state = np.array(self.nodes + self.tasks)
        self.done = False
        return self.state

    def step(self, action):
        task_idx, node_idx = action
        reward = 0
        if self.state[node_idx] >= self.state[len(self.nodes) + task_idx]:
            self.state[node_idx] -= self.state[len(self.nodes) + task_idx]
            reward = 1
            self.state[len(self.nodes) + task_idx] = 0
        else:
            reward = -1
            # self.done = True
        if np.sum(self.state[len(self.nodes):]) == 0:
            self.done = True
        return self.state, reward, self.done

# DQN模型
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
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# 选择动作
def select_action(state, policy_net, steps_done, n_actions):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(torch.tensor([state], dtype=torch.float)).max(1)[1].view(1, 1), steps_done
    else:
        return torch.tensor([[random.randrange(n_actions)]], dtype=torch.int64), steps_done

# 优化模型
def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    experiences = memory.sample(BATCH_SIZE)
    batch = Experience(*zip(*experiences))

    state_batch = torch.tensor(np.array(batch.state), dtype=torch.float)
    action_batch = torch.tensor(batch.action).unsqueeze(-1)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float)
    non_final_next_states = torch.tensor([s for s in batch.next_state if s is not None], dtype=torch.float)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values[non_final_next_states] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

# 训练循环参数
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
BATCH_SIZE = 128
LR = 0.001

# 环境和模型初始化
env = AllocationEnv([2, 1, 3], [3, 4, 2])  # 示例：3个任务和3个节点
n_actions = len(env.tasks) * len(env.nodes)
policy_net = DQN(len(env.state), n_actions)
target_net = DQN(len(env.state), n_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayMemory(10000)
steps_done = 0

# 训练循环
num_episodes = 50
for i_episode in range(num_episodes):
    state = env.reset()
    for t in count():
        action, steps_done = select_action(state, policy_net, steps_done, n_actions)
        next_state, reward, done = env.step(action.item())
        memory.push(state, action, next_state, reward)
        state = next_state

        optimize_model(memory, policy_net, target_net, optimizer)

        if done:
            break

    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print("Training complete")
