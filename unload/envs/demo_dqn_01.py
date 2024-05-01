import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple

# Define the neural network architecture
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define experience replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Define the DQN agent
class DQNAgent:
    def __init__(self, input_dim, output_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.policy_net = DQN(input_dim, output_dim)
        self.target_net = DQN(input_dim, output_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(capacity=10000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.output_dim - 1)
        else:
            with torch.no_grad():
                q_values = self.policy_net(torch.tensor(state, dtype=torch.float32))
                return q_values.argmax().item()

    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.tensor([s for s in batch.next_state if s is not None], dtype=torch.float32)
        state_batch = torch.tensor(batch.state, dtype=torch.float32)
        action_batch = torch.tensor(batch.action, dtype=torch.int64)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        next_state_values = torch.zeros(batch_size, dtype=torch.float32)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = reward_batch + self.gamma * next_state_values

        loss = nn.functional.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

'''
# Define the environment
class ResourceAllocationEnv:
    def __init__(self, num_tasks, task_resources, num_nodes, node_resources):
        self.num_tasks = num_tasks
        self.task_resources = task_resources
        self.num_nodes = num_nodes
        self.node_resources = node_resources
        self.state = np.array(self.node_resources + self.task_resources)

    def reset(self):
        self.node_resources = np.random.randint(1, 10, size=self.num_nodes)  # Reinitialize node resources
        self.state = np.array(self.node_resources + self.task_resources)
        return self.state

    def step(self, action):
        node_index, task_index = action
        if self.node_resources[node_index] >= self.task_resources[task_index]:
            self.node_resources[node_index] -= self.task_resources[task_index]
            self.task_resources[task_index] = 0
        else:
            self.task_resources[task_index] -= self.node_resources[node_index]
            self.node_resources[node_index] = 0

        reward = self._calculate_reward()
        done = all(task == 0 for task in self.task_resources)
        self.state = np.array(self.node_resources + self.task_resources)
        return self.state, reward, done, {}

    def _calculate_reward(self):
        return np.mean(self.node_resources) / np.sum(self.node_resources)

    def get_valid_actions(self):
        valid_actions = []
        for i in range(self.num_nodes):
            for j in range(self.num_tasks):
                if self.node_resources[i] >= self.task_resources[j]:
                    valid_actions.append((i, j))
        return valid_actions
'''


# Define the environment
class ResourceAllocationEnv:
    def __init__(self, num_tasks, task_resources, num_nodes, node_resources):
        self.num_tasks = num_tasks
        self.task_resources = task_resources
        self.num_nodes = num_nodes
        self.node_resources = node_resources
        self.state = self._get_state()

    def reset(self):
        self.node_resources = np.random.randint(1, 10, size=self.num_nodes)  # Reinitialize node resources
        self.state = self._get_state()
        return self.state

    def step(self, action):
        node_index, task_index = action
        if self.node_resources[node_index] >= self.task_resources[task_index]:
            self.node_resources[node_index] -= self.task_resources[task_index]
            self.task_resources[task_index] = 0
        else:
            self.task_resources[task_index] -= self.node_resources[node_index]
            self.node_resources[node_index] = 0

        reward = self._calculate_reward()
        done = all(task == 0 for task in self.task_resources)
        self.state = self._get_state()
        return self.state, reward, done, {}

    def _calculate_reward(self):
        return np.mean(self.node_resources) / np.sum(self.node_resources)

    def _get_state(self):
        state = np.concatenate((self.node_resources, self.task_resources))
        return state

    def get_valid_actions(self):
        valid_actions = []
        for i in range(self.num_nodes):
            for j in range(self.num_tasks):
                if self.node_resources[i] >= self.task_resources[j]:
                    valid_actions.append((i, j))
        return valid_actions


# Initialize environment and agent
env = ResourceAllocationEnv(num_tasks=3, task_resources=[2, 3, 4], num_nodes=2, node_resources=[5, 6])
input_dim = len(env.reset())
output_dim = len(env.get_valid_actions())
agent = DQNAgent(input_dim, output_dim)

# Train the agent
num_episodes = 1000
batch_size = 32
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action_index = agent.select_action(state)
        action = env.get_valid_actions()[action_index]
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        agent.memory.push(state, action_index, next_state if not done else None, reward, done)
        state = next_state
        agent.train(batch_size)
    agent.update_target_network()
    print(f"Episode {episode+1}, Total Reward: {total_reward}")

# Test the agent
state = env.reset()
done = False
while not done:
    action_index = agent.select_action(state)
    action = env.get_valid_actions()[action_index]
    next_state, reward, done, _ = env.step(action)
    state = next_state
    print(f"Action: {action}, Reward: {reward}, Done: {done}")

