import gym
from gym import spaces
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class TaskOffloadingEnv(gym.Env):
    def __init__(self, task_demands, node_resources):
        super(TaskOffloadingEnv, self).__init__()
        self.task_demands = task_demands
        self.node_resources = node_resources
        self.n_tasks = len(task_demands)
        self.m_nodes = len(node_resources)
        self.action_space = spaces.MultiDiscrete([self.m_nodes] * self.n_tasks)
        self.observation_space = spaces.Box(low=0, high=np.inf,
                                            shape=(self.n_tasks + self.m_nodes,), dtype=np.float32)
        self.state = None
        self.reset()

    def step(self, actions):
        assert len(actions) == self.n_tasks, "Action dimension mismatch."
        reward = 0
        done = False
        info = {}

        # Update node resources based on actions
        for i, action in enumerate(actions):
            if self.node_resources[action] >= self.task_demands[i]:
                self.node_resources[action] -= self.task_demands[i]  # 更新节点剩余资源
                reward += self.task_demands[i]  # Simple reward: successful resource allocation
            else:
                reward -= self.task_demands[i]  # Penalty for failing to allocate

        self.state = np.concatenate((self.task_demands, self.node_resources))

        # Check if all tasks have been allocated or not
        if all(res == 0 for res in self.node_resources):
            done = True

        return self.state, reward, done, info

    def reset(self):
        self.state = np.concatenate((self.task_demands, self.node_resources))
        return self.state

    def render(self, mode='human'):
        pass  # Optional: Implement visualization if needed



class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)


def train_dqn(env, dqn, episodes, learning_rate=0.01):
    optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = env.action_space.sample()  # Random action for simplicity
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Convert to torch tensors
            state_t = torch.FloatTensor(state)
            next_state_t = torch.FloatTensor(next_state)
            reward_t = torch.FloatTensor([reward])
            action_t = torch.LongTensor(action)

            # Compute loss and update DQN
            current_q_values = dqn(state_t)
            max_next_q_values = dqn(next_state_t).detach().max()
            expected_q_values = reward_t + (0.99 * max_next_q_values)

            loss = loss_fn(current_q_values, expected_q_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")


# 初始化环境和DQN
task_demands = [1, 9, 2, 7, 4, 3, 2, 5, 6, 1]
node_resources = [10, 15, 30]
env = TaskOffloadingEnv(task_demands, node_resources)
input_size = env.observation_space.shape[0]
hidden_size = 128  # 可以调整
output_size = env.action_space.nvec.prod()  # 总动作空间大小
dqn = DQN(input_size, hidden_size, output_size)

# 训练
episodes = 200  # 可以调整
train_dqn(env, dqn, episodes)


