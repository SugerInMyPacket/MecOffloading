import numpy as np
import gym
from gym import spaces

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class TaskAllocationEnv(gym.Env):
    def __init__(self, num_tasks, num_nodes, node_resources):
        super(TaskAllocationEnv, self).__init__()
        self.num_tasks = num_tasks
        self.num_nodes = num_nodes
        self.node_resources = np.array(node_resources, dtype=np.int32)
        self.action_space = spaces.Discrete(self.num_nodes)
        self.observation_space = spaces.Box(low=0, high=np.max(node_resources), shape=(self.num_nodes + 1,), dtype=np.int32)
        self.current_task = 0
        self.task_resource_need = None

    def reset(self):
        self.node_resources = np.random.randint(1, 10, size=self.num_nodes)
        self.current_task = 0
        self.task_resource_need = np.random.randint(1, 10)
        return np.append(self.node_resources, self.task_resource_need)

    def step(self, action):
        done = False
        reward = 0

        if self.node_resources[action] >= self.task_resource_need:
            self.node_resources[action] -= self.task_resource_need
            reward = 1  # Simple reward, can be adjusted
            self.current_task += 1
        else:
            reward = -10  # Penalty for not being able to allocate

        if self.current_task >= self.num_tasks:
            done = True

        self.task_resource_need = np.random.randint(1, 10) if not done else 0
        next_state = np.append(self.node_resources, self.task_resource_need)
        return next_state, reward, done, {}

    def render(self, mode='human'):
        pass  # For simplicity, we won't implement visualization



class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)

def select_action(state, model, epsilon, num_actions):
    if random.random() > epsilon:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_value = model(state)
            action = q_value.max(1)[1].item()
    else:
        action = random.randrange(num_actions)
    return action

def optimize_model(batch_size, replay_buffer, model, target_model, optimizer, gamma):
    if len(replay_buffer) < batch_size:
        return
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    state = torch.FloatTensor(state)
    next_state = torch.FloatTensor(next_state)
    action = torch.LongTensor(action)
    reward = torch.FloatTensor(reward)
    done = torch.FloatTensor(done)

    q_values = model(state).gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_values = target_model(next_state).max(1)[0]
    expected_q_values = reward + gamma * next_q_values * (1 - done)

    loss = nn.MSELoss()(q_values, expected_q_values.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

if __name__ == '__main__':
    # 参数设置
    num_tasks = 10
    task_resource_need = [1, 5, 9, 7, 8, 6, 4, 3, 2, 1]
    num_nodes = 3
    node_resources = [10, 20, 30]
    num_episodes = 200
    batch_size = 32
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500
    target_update = 10

    # 环境和模型初始化
    env = TaskAllocationEnv(num_tasks=10, num_nodes=3, node_resources=[10, 20, 30])
    n_actions = env.action_space.n
    model = DQN(env.observation_space.shape[0], n_actions)
    target_model = DQN(env.observation_space.shape[0], n_actions)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters())
    replay_buffer = ReplayBuffer(10000)

    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * np.exp(
        -1. * frame_idx / epsilon_decay)

    # 训练循环
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        for t in range(num_tasks):
            epsilon = epsilon_by_frame(episode)
            action = select_action(state, model, epsilon, n_actions)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            optimize_model(batch_size, replay_buffer, model, target_model, optimizer, gamma)

            if done:
                break

        if episode % target_update == 0:
            target_model.load_state_dict(model.state_dict())

        print('Total reward in episode {}: {}'.format(episode, total_reward))


