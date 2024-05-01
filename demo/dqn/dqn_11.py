import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple, deque

class TaskAllocationEnv:
    def __init__(self, tasks, nodes):
        self.tasks = tasks  # 任务资源需求列表
        self.nodes = nodes  # 节点资源列表
        self.n_tasks = len(tasks)
        self.n_nodes = len(nodes)
        self.state = np.array(nodes)

    def reset(self):
        self.state = np.array(self.nodes)
        return self.state

    def step(self, action):
        task_id, node_id = action
        reward = 0
        done = False

        # 如果节点有足够的资源执行任务
        if self.state[node_id] >= self.tasks[task_id]:
            self.state[node_id] -= self.tasks[task_id]
            reward = 1  # 给予正奖励
        else:
            reward = -1  # 资源不足，给予负奖励

        # 检查是否所有任务都已经分配
        if np.all(self.state < np.min(self.tasks)):
            done = True

        return self.state, reward, done

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

class DQN(nn.Module):
    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(inputs, 50)
        self.fc2 = nn.Linear(50, outputs)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size, batch_size=128, gamma=0.99, lr=0.001, memory_size=10000, target_update=10):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.memory = ReplayMemory(memory_size)
        self.policy_net = DQN(state_size, action_size).float()
        self.target_net = DQN(state_size, action_size).float()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 不计算梯度
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.steps_done = 0
        self.target_update = target_update

    def select_action(self, state, epsilon):
        sample = random.random()
        if sample > epsilon:
            with torch.no_grad():
                return self.policy_net(torch.from_numpy(state).float().unsqueeze(0)).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_size)]], dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def train(env, agent, episodes, epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=200):
    for episode in range(episodes):
        state = env.reset()
        for t in count():
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                      math.exp(-1. * agent.steps_done / epsilon_decay)
            action = agent.select_action(state, epsilon)
            next_state, reward, done = env.step(action.item())

            reward = torch.tensor([reward], dtype=torch.float)

            if not done:
                next_state = None

            agent.memory.push(state, action, next_state, reward)
            state = next_state

            agent.optimize_model()
            if done:
                break

            if episode % agent.target_update == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())

    print('Complete')
    torch.save(agent.policy_net.state_dict(), '/mnt/data/dqn_task_allocation.pth')

    # 保存模型
    torch.save(agent.policy_net.state_dict(), '/mnt/data/dqn_task_allocation.pth')

# 加载模型
model = DQN(state_size, action_size)
model.load_state_dict(torch.load('/mnt/data/dqn_task_allocation.pth'))
model.eval()



