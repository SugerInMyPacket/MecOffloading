import numpy as np


class TaskAllocationEnvironment:
    # 初始化环境
    def __init__(self, num_tasks, num_nodes, node_resources, task_resources):
        self.num_tasks = num_tasks
        self.num_nodes = num_nodes
        self.node_resources = node_resources
        self.task_resources = task_resources
        self.state_size = num_nodes
        self.action_size = num_tasks * num_nodes

    # 重置
    def reset(self):
        self.node_states = np.array(self.node_resources)  # 初始节点资源
        self.task_states = np.array(self.task_resources)  # 初始任务资源需求
        self.done = False
        return self.node_states

    def step(self, action):
        task_id = action // self.num_nodes
        node_id = action % self.num_nodes

        if self.task_states[task_id] <= self.node_states[node_id]:
            self.node_states[node_id] -= self.task_states[task_id]
            self.task_states[task_id] = 0
        # else:
        #     self.task_states[task_id] -= self.node_states[node_id]
        #     self.node_states[node_id] = 0

        # reward = -np.sum(self.node_states)  # 惩罚node剩余资源
        reward = -np.sum(self.task_states)  # 惩罚node剩余资源
        if np.sum(self.task_states) == 0:
            self.done = True  # 所有任务完成

        return self.node_states, reward, self.done

    def get_state_size(self):
        return self.state_size

    def get_action_size(self):
        return self.action_size


# Q-learning 代理
class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.99,
                 exploration_rate=1.0, exploration_decay=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(self.action_size)
        return np.argmax(self.q_table[state, :])

    def learn(self, state, action, reward, next_state, done):
        q_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state, :]) if not done else 0
        target = reward + self.discount_factor * next_max
        self.q_table[state, action] += self.learning_rate * (target - q_value)
        if done:
            self.exploration_rate *= self.exploration_decay


# 训练代理
def train_agent(env, agent, episodes=500):
    for episode in range(episodes):
        state = env.reset()
        state = np.argmax(state)  # 转换节点状态为标量以用于 Q 表
        total_reward = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            next_state = np.argmax(next_state)  # 转换节点状态为标量以用于 Q 表
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")


# 示例用法
if __name__ == "__main__":
    num_tasks = 10
    task_resources = [3, 5, 7, 1, 4, 5, 6, 9, 8, 7]
    num_nodes = 3
    node_resources = [10, 20, 50]

    env = TaskAllocationEnvironment(num_tasks, num_nodes, node_resources, task_resources)
    agent = QLearningAgent(env.get_state_size(), env.get_action_size())
    train_agent(env, agent)
