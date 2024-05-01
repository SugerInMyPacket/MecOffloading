import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

# __init__方法初始化环境,接收任务数量、节点数量、每个任务所需的资源量和每个节点的初始资源量。
# reset方法重置环境状态,包括剩余任务列表、节点资源量和任务分配情况。它返回初始状态。
# get_state方法返回当前状态,包括剩余任务列表、节点资源量和任务分配情况。
# step方法执行给定的动作(分配任务给节点)。如果该动作不合法(任务已分配或节点资源不足),则返回负奖励。
#   否则,它更新环境状态并返回新状态、奖励和是否完成的标志。
# render方法打印当前环境状态,用于调试和可视化。
class TaskAllocationEnv:
    def __init__(self, num_tasks, num_nodes, task_resources, node_capacities):
        self.num_tasks = num_tasks
        self.num_nodes = num_nodes
        self.task_resources = task_resources
        self.node_capacities = node_capacities
        self.reset()

    def reset(self):
        self.remaining_tasks = list(range(self.num_tasks))
        self.node_resources = self.node_capacities.copy()
        self.task_allocations = [-1] * self.num_tasks
        return self.get_state()

    def get_state(self):
        return np.array([self.remaining_tasks, self.node_resources, self.task_allocations])
        # return np.concatenate(self.remaining_tasks, self.node_resources, self.task_allocations)

    def step(self, action):
        task_idx, node_idx = action
        task_resource = self.task_resources[task_idx]
        node_resource = self.node_resources[node_idx]

        if task_idx not in self.remaining_tasks or node_resource < task_resource:
            return self.get_state(), -1, False

        self.remaining_tasks.remove(task_idx)
        self.node_resources[node_idx] -= task_resource
        self.task_allocations[task_idx] = node_idx

        reward = 1 if len(self.remaining_tasks) == 0 else 0
        done = len(self.remaining_tasks) == 0

        return self.get_state(), reward, done

    def render(self):
        print(f"Remaining tasks: {self.remaining_tasks}")
        print(f"Node resources: {self.node_resources}")
        print(f"Task allocations: {self.task_allocations}")


# 初始化任务分配环境TaskAllocationEnv。
# 创建DQN智能体DQNAgent。
# 进行训练循环,每个循环包含多个时间步骤:
#   在每个时间步骤,智能体根据当前状态选择一个动作(分配任务给节点)。
#   执行选择的动作,获取下一个状态、奖励和是否完成的标志。
#   将转换存储在经验回放缓冲区中。
#   如果缓冲区足够大,从中采样一批数据进行经验回放,更新Q网络。
#   如果所有任务都已分配,打印本轮的总奖励并进入下一个循环。
#   每100个训练循环保存一次模型权重。

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(1, self.state_size)))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model

    def memorize(self, state, action, reward, next_state, done):
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
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

# 使用DQN解决任务分配问题
env = TaskAllocationEnv(num_tasks=10,
                        num_nodes=3,
                        task_resources=[2, 3, 1, 4, 2, 1, 3, 2, 1, 2],
                        node_capacities=[5, 6, 4]
                        )

state_size = env.get_state().shape[1]
action_size = env.num_tasks * env.num_nodes
agent = DQNAgent(state_size, action_size)

batch_size = 32
episodes = 1000

for e in range(episodes):
    state = env.reset()
    state = state.flatten()
    state = np.reshape(state, [1, state_size])
    for time in range(500):
        action = agent.act(state)
        task_idx = action // env.num_nodes
        node_idx = action % env.num_nodes
        next_state, reward, done = env.step((task_idx, node_idx))
        next_state = next_state.flatten()
        next_state = np.reshape(next_state, [1, state_size])
        agent.memorize(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"Episode: {e}/{episodes}, Reward: {reward}")
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    if e % 100 == 0:
        agent.save(f"dqn_model_{e}.h5")

