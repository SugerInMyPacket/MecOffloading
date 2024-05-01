import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random


class TaskAllocationEnv:
    def __init__(self, tasks, nodes):
        self.tasks = np.array(tasks)  # 任务所需资源量
        self.nodes = np.array(nodes)  # 节点总资源量
        self.n_tasks = len(tasks)
        self.n_nodes = len(nodes)
        self.state = None
        self.reset()

    def reset(self):
        # 重置环境到初始状态
        self.state = np.concatenate((self.tasks, self.nodes))
        return self.state

    def step(self, action):
        task_id = action // self.n_nodes  # 从动作中获取任务ID
        node_id = action % self.n_nodes   # 从动作中获取节点ID
        done = False
        reward = 0

        # 检查动作是否有效（节点是否有足够资源）
        if self.tasks[task_id] <= self.nodes[node_id]:
            self.nodes[node_id] -= self.tasks[task_id]
            self.tasks[task_id] = 0
            reward = 1  # 成功分配任务奖励为1
        else:
            reward = -1  # 任务不能被分配，给予惩罚

        if np.sum(self.tasks) == 0:
            done = True  # 所有任务都被成功分配，结束

        self.state = np.concatenate((self.tasks, self.nodes))
        return self.state, reward, done

    def render(self):
        # 可选：实现一个方法来可视化或打印当前环境状态
        print(f"Tasks: {self.tasks}, Nodes: {self.nodes}")



class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state.reshape(1, -1))
        return np.argmax(act_values[0])  # returns the action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape(1, -1))[0])
            target_f = self.model.predict(state.reshape(1, -1))
            target_f[0][action] = target
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# 初始化环境和代理
env = TaskAllocationEnv([2, 3, 4], [5, 5, 5])  # 示例：3个任务和3个节点
agent = DQNAgent(env.n_tasks + env.n_nodes, env.n_tasks * env.n_nodes)

# 训练代理
episodes = 1000
for e in range(episodes):
    state = env.reset()
    for time in range(500):
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"episode: {e+1}/{episodes}, score: {time}, epsilon: {agent.epsilon:.2}")
            break
        if len(agent.memory) > 32:
            agent.replay(32)


