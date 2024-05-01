import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

# 环境类
class Env:
    def __init__(self, M, init_resources, N):
        self.M = M
        self.init_resources = init_resources
        self.N = N
        self.reward = 0
        # self.done = False

    def reset(self):
        self.nodes = np.array(self.init_resources)
        self.task_count = 0
        return self.nodes

    def step(self, action, task_resource):
        reward = 0
        done = False

        if self.nodes[action] >= task_resource:
            self.nodes[action] -= task_resource
            # self.reward += 1
            reward = 1
            self.task_count += 1
        else:
            reward = -5
            # self.reward -= 5

        # 任务全部分配
        if self.task_count == self.N:
            done = True

        return self.nodes, reward, done

# DQN类
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        return self.fc(x)


# 经验回放缓冲区类
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, state, action, reward, next_state):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Agent类
# 修改Agent类，使用经验回放缓冲区
# 修改Agent类，添加目标网络
class Agent:
    def __init__(self, M, N, init_resources, learning_rate=0.01, gamma=0.99, buffer_capacity=1000, batch_size=64, target_update=10):
        self.env = Env(M, init_resources, N)
        self.dqn = DQN(M, M)
        self.target_dqn = DQN(M, M)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.target_dqn.eval()
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.target_update = target_update

    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state)
        actions = self.dqn(state_tensor)
        return torch.argmax(actions).item()

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        state_tensor = torch.FloatTensor(states)
        action_tensor = torch.tensor(actions)
        reward_tensor = torch.FloatTensor(rewards)
        next_state_tensor = torch.FloatTensor(next_states)

        q_values = self.dqn(state_tensor)
        next_q_values = self.target_dqn(next_state_tensor)
        q_value = q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        next_q_value = torch.max(next_q_values, 1)[0]
        expected_q_value = reward_tensor + self.gamma * next_q_value

        loss = self.loss_fn(q_value, expected_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())

# 修改训练函数，使用经验回放缓冲区
def train(M, N, init_resources, tasks, num_episodes, learning_rate=0.01, gamma=0.99, buffer_capacity=1000, batch_size=64):
    agent = Agent(M, N, init_resources, learning_rate, gamma, buffer_capacity, batch_size)

    for episode in range(num_episodes):
        state = agent.env.reset()
        total_reward = 0

        for task_resource in tasks:
            action = agent.choose_action(state)
            next_state, reward, done = agent.env.step(action, task_resource)
            agent.replay_buffer.push(state, action, reward, next_state)
            agent.learn()
            total_reward += reward
            state = next_state

            if done:
                break

        print(f"Episode {episode}: Total reward: {total_reward}")


# 保存模型
def save_model(agent, model_path):
    torch.save(agent.dqn.state_dict(), model_path)

# 加载模型
def load_model(agent, model_path):
    if os.path.isfile(model_path):
        agent.dqn.load_state_dict(torch.load(model_path))
        agent.dqn.eval()
        agent.update_target_network()

# 修改训练函数，支持加载和保存模型
def train(M, N, init_resources, tasks, num_episodes, model_path, learning_rate=0.01, gamma=0.99, buffer_capacity=1000, batch_size=64, target_update=10):
    agent = Agent(M, N, init_resources, learning_rate, gamma, buffer_capacity, batch_size, target_update)

    # 加载模型（如果存在）
    load_model(agent, model_path)

    for episode in range(num_episodes):
        state = agent.env.reset()
        total_reward = 0

        for task_resource in tasks:
            action = agent.choose_action(state)
            next_state, reward, done = agent.env.step(action, task_resource)
            agent.replay_buffer.push(state, action, reward, next_state)
            agent.learn()
            total_reward += reward
            state = next_state

            if done:
                break

        if episode % target_update == 0:
            agent.update_target_network()

        print(f"Episode {episode}: Total reward: {total_reward}")

    # 保存模型
    save_model(agent, model_path)


# 示例
M = 5
init_resources = [10, 10, 10, 10, 10]
N = 10
tasks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
num_episodes = 200
model_path = "model.pt"

train(M, N, init_resources, tasks, num_episodes, model_path)