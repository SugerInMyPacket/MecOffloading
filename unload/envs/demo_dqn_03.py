import gym
import numpy as np
from collections import deque
import random
import tensorflow as tf
from tensorflow.python import keras
from keras import models
from keras import layers
from keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras import optimizers


class TaskAllocationEnv(gym.Env):
    def __init__(self, num_tasks, num_nodes, node_resources):
        self.num_tasks = num_tasks
        self.num_nodes = num_nodes
        self.node_resources = node_resources
        self.task_resources = np.random.randint(1, 11, size=num_tasks)

        self.action_space = gym.spaces.Discrete(num_nodes)
        self.observation_space = gym.spaces.MultiDiscrete([num_nodes] * num_tasks)

        self.state = np.zeros(num_tasks, dtype=int)
        self.reward = 0
        self.done = False

    def step(self, action):
        # 将任务分配到指定节点
        node = action
        task = self.state.index(0)
        if self.node_resources[node] >= self.task_resources[task]:
            self.state[task] = node
            self.node_resources[node] -= self.task_resources[task]
            self.reward += 1
        else:
            self.reward -= 1

        # 检查是否所有任务都已分配
        if np.all(self.state != 0):
            self.done = True

        return self.state, self.reward, self.done, {}

    def reset(self):
        self.state = np.zeros(self.num_tasks, dtype=int)
        self.reward = 0
        self.done = False
        self.node_resources = np.copy(self.node_resources)
        return self.state


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(24, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.optimizers.Adam(learning_rate=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_dqn(env, agent, episodes=1000, batch_size=32):
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.num_tasks])
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, env.num_tasks])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("Episode {}/{}, score: {}, e: {:.2}".format(e, episodes, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

def main():
    # 设置环境参数
    num_tasks = 10
    num_nodes = 5
    node_resources = np.array([50, 40, 60, 30, 45])

    # 创建环境和智能体
    env = TaskAllocationEnv(num_tasks, num_nodes, node_resources)
    agent = DQNAgent(env.num_tasks, env.num_nodes)

    # 训练智能体
    train_dqn(env, agent)

    # 测试训练好的模型
    state = env.reset()
    done = False
    while not done:
        action = agent.act(np.reshape(state, [1, env.num_tasks]))
        state, reward, done, _ = env.step(action)
        print("State:", state, "Action:", action, "Reward:", reward)

if __name__ == "__main__":
    main()

