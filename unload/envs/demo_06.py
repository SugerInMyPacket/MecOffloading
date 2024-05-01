import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque
import matplotlib.pyplot as plt

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# DDQN 网络结构
class DDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DDQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# 经验回放
class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


# DDQN Agent
class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = 100000
        self.batch_size = 64
        self.gamma = 0.99  # discount factor
        self.lr = 0.0005  # learning rate
        self.tau = 0.001  # for soft update of target parameters

        # DDQN 网络
        self.policy_net = DDQN(state_size, action_size).to(device)
        self.target_net = DDQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size)

        # 确保两个网络具有相同的初始权重
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def act(self, state, eps=0.01):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        self.policy_net.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        Q_expected = self.policy_net(states).gather(1, actions)

        loss = nn.MSELoss()(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.policy_net, self.target_net, self.tau)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ExtendedEdgeEnv:
    def __init__(self):
        self.num_servers = 10
        self.server_loads = np.zeros(self.num_servers)  # 服务器当前负载
        self.server_queues = [[] for _ in range(self.num_servers)]  # 服务器任务队列
        self.state_size = self.num_servers + 4  # 服务器负载 + 任务大小 + 预期完成时间 + 任务类型 + 优先级
        self.action_size = self.num_servers  # 选择的服务器
        self.state = self.generate_state()  # 初始化状态

    def reset(self):
        self.server_loads = np.zeros(self.num_servers)
        self.server_queues = [[] for _ in range(self.num_servers)]
        self.state = self.generate_state()
        return self.state

    def generate_state(self):
        task_size = np.random.rand()  # 任务大小
        task_time = np.random.rand()  # 任务预期完成时间
        task_type = np.random.randint(0, 3)  # 任务类型，例如0, 1, 2
        task_priority = np.random.randint(1, 6)  # 任务优先级，例如1-5
        return np.concatenate([self.server_loads, [task_size, task_time, task_type, task_priority]])

    def step(self, action):
        task = self.state[-4:]
        self.server_queues[action].append(task)
        self.update_loads()

        next_state = self.generate_state()
        reward = self.calculate_reward(action, task)
        done = np.random.choice([True, False])
        return next_state, reward, done, {}

    def update_loads(self):
        # 更新服务器负载和处理任务
        for i, queue in enumerate(self.server_queues):
            if queue and random.random() < 0.5:  # 假设有一定概率完成任务
                completed_task = queue.pop(0)
                self.server_loads[i] -= completed_task[0]  # 减去完成任务的大小

    def calculate_reward(self, action, task):
        task_size, task_time, task_type, task_priority = task
        time_penalty = task_time
        energy_penalty = self.server_loads[action]
        load_penalty = np.mean(self.server_loads)
        priority_reward = task_priority  # 高优先级任务给予更高奖励
        return priority_reward - (time_penalty + energy_penalty + load_penalty)


# 主程序
def main():
    env = ExtendedEdgeEnv()
    agent = DDQNAgent(env.state_size, env.action_size)

    num_episodes = 500
    max_t = 200
    rewards_per_episode = []
    steps_per_episode = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                rewards_per_episode.append(total_reward)
                steps_per_episode.append(t + 1)
                break

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards_per_episode)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    plt.subplot(1, 2, 2)
    plt.plot(steps_per_episode)
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()