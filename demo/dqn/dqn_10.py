import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class TaskAllocationEnv:
    def __init__(self, tasks, nodes):
        self.tasks = tasks # List of task resource requirements
        self.nodes = nodes # List of node total resources
        self.state = None
        self.task_index = 0

    def reset(self):
        self.state = np.array(self.nodes)
        self.task_index = 0
        return self.state, self.task_index

    def step(self, action):
        task_requirement = self.tasks[self.task_index]
        reward = 0
        done = False

        if self.state[action] >= task_requirement:
            self.state[action] -= task_requirement
            reward = 1 # Successfully allocated
            self.task_index += 1
        else:
            reward = -1 # Failed to allocate

        # self.task_index += 1
        if self.task_index >= len(self.tasks):
            done = True

        return self.state, reward, done, self.task_index



class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99 # discount factor
        self.epsilon = 1.0 # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # epsilon-贪婪策略采取动作
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state_tensor).detach().numpy()
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                target = (reward + self.gamma *
                          np.amax(self.model(next_state_tensor).detach().numpy()))
            target_f = self.model(torch.FloatTensor(state).unsqueeze(0)).detach().numpy()
            target_f[0][action] = target
            self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
            # 计算损失
            loss = nn.MSELoss()(self.model(torch.FloatTensor(state).unsqueeze(0)), torch.FloatTensor(target_f))
            loss.backward()  # 反向传播更新参数
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))


def train(env, agent, episodes, batch_size):

    for e in range(episodes):
        total_reward = 0
        state, _ = env.reset()
        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        print(f"Episode {e+1}/{episodes} completed, Total Reward: {total_reward}")
    agent.save("dqn_10_model.pth")


# 参数初始化
tasks = [2, 3, 4, 1, 9, 6] # Example tasks
nodes = [5, 10, 15] # Example nodes
env = TaskAllocationEnv(tasks, nodes)
agent = DQNAgent(len(nodes), len(nodes))
episodes = 200
batch_size = 32

# 开始训练
train(env, agent, episodes, batch_size)

