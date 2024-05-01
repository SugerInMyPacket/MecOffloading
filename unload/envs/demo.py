import numpy as np
import cv2
from PIL import Image
import time
import pickle
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

SIZE = 10  # 游戏区域的大小
EPISODES = 30000  # 局数
SHOW_EVERY = 3000  # 定义每隔多少局展示一次图像

FOOD_REWARD = 25  # agent获得食物的奖励
ENEMY_PENALITY = 300  # 遇上对手的惩罚
MOVE_PENALITY = 1  # 每移动一步的惩罚

epsilon = 0.6
EPS_DECAY = 0.9998
DISCOUNT = 0.95
LEARNING_RATE = 0.1

q_table = None
# 设定三个部分的颜色分别是蓝、绿、红
d = {1: (255, 0, 0),  # blue
     2: (0, 255, 0),  # green
     3: (0, 0, 255)}  # red
PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3


# 智能体的类，有其 位置信息 和 动作函数
class Cube:
    def __init__(self):  # 随机初始化位置坐标
        self.x = np.random.randint(0, SIZE - 1)
        self.y = np.random.randint(0, SIZE - 1)

    def __str__(self):
        return f'{self.x},{self.y}'

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def action(self, choise):
        if choise == 0:
            self.move(x=1, y=1)
        elif choise == 1:
            self.move(x=-1, y=1)
        elif choise == 2:
            self.move(x=1, y=-1)
        elif choise == 3:
            self.move(x=-1, y=-1)

    def move(self, x=False, y=False):
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        if self.x < 0:
            self.x = 0
        if self.x > SIZE - 1:
            self.x = SIZE - 1
        if self.y < 0:
            self.y = 0
        if self.y > SIZE - 1:
            self.y = SIZE - 1


# 初始化Q表格
if q_table is None:  # 如果没有实现提供，就随机初始化一个Q表格
    q_table = {}
    for x1 in range(-SIZE + 1, SIZE):
        for y1 in range(-SIZE + 1, SIZE):
            for x2 in range(-SIZE + 1, SIZE):
                for y2 in range(-SIZE + 1, SIZE):
                    q_table[((x1, y1), (x2, y2))] = [np.random.randint(-5, 0) for i in range(4)]
else:  # 提供了，就使用提供的Q表格
    with open(q_table, 'rb') as f:
        q_table = pickle.load(f)


# 训练一个智能体
episode_rewards = []  # 初始化奖励序列
for episode in range(EPISODES):
    # 实例化玩家、食物和敌人
    player = Cube()
    food = Cube()
    enemy = Cube()

    # 每隔一段时间设定show为True，显示图像
    if episode % SHOW_EVERY == 0:
        print('episode ', episode, '  epsilon:', epsilon)
        print('mean_reward:', np.mean(episode_rewards[-SHOW_EVERY:]))
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(200):
        obs = (player - food, player - enemy)  # 观测
        # 开发和探索并存
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])  # 选择Q值最高的动作，来进行开发
        else:
            action = np.random.randint(0, 4)  # 随机选择一个动作，进行探索

        # print("player的位置：",player)
        # print("player的观测：",obs)
        # print("player的动作：",action)
        player.action(action)  # 智能体执行动作
        # food.move()
        # enemy.move()

        # print("player的下一步位置：",player)

        # 奖励
        if player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        elif player.x == enemy.x and player.y == enemy.y:
            reward = - ENEMY_PENALITY
        else:
            reward = - MOVE_PENALITY

        # print('reward:',reward)

        # 更新Q表格
        current_q = q_table[obs][action]  # 当前动作、状态对应的Q值
        # print('current_q:',current_q)
        new_obs = (player - food, player - enemy)  # 动作之后新的状态
        # print('new_obs:',new_obs)
        max_future_q = np.max(q_table[new_obs])  # 新的状态下，最大的Q值
        # print('max_future_q:',max_future_q)
        # print('')
        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[obs][action] = new_q

        # 图像显示
        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            env[food.x][food.y] = d[FOOD_N]
            env[player.x][player.y] = d[PLAYER_N]
            env[enemy.x][enemy.y] = d[ENEMY_N]
            img = Image.fromarray(env, 'RGB')
            img = img.resize((400, 400))
            cv2.imshow('', np.array(img))
            if reward == FOOD_REWARD or reward == -ENEMY_PENALITY:
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        #
        episode_reward += reward

        if reward == FOOD_REWARD or reward == ENEMY_PENALITY:
            break

    # Note:
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.xlabel('episode #')
plt.ylabel(f'mean{SHOW_EVERY} reward')
plt.show()

with open(f'qtable_{int(time.time())}.pickle', 'wb') as f:
    pickle.dump(q_table, f)
