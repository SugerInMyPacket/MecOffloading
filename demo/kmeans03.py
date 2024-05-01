import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 生成随机的任务数据，包括任务大小、CPU计算密度和截止时间
np.random.seed(0)
num_tasks = 100
task_sizes = np.random.randint(1, 100, num_tasks)
cpu_densities = np.random.uniform(0.5, 2.0, num_tasks)
deadlines = np.random.randint(1, 50, num_tasks)

# 计算每个任务的计算量（任务大小 * CPU计算密度）
computational_loads = task_sizes * cpu_densities

# 将任务属性组合成特征矩阵
features = np.column_stack((computational_loads, deadlines))

# 设定四个中心点的位置
centers = np.array([[500, 10], [500, 40], [50, 10], [50, 40]])

# 使用K-means聚类算法
kmeans = KMeans(n_clusters=4, init=centers, n_init=1, random_state=0)
kmeans.fit(features)
labels = kmeans.labels_

# 可视化聚类结果
plt.figure(figsize=(8, 6))

plt.scatter(features[labels == 0, 0], features[labels == 0, 1], s=50, c='red', label='Category 1')
plt.scatter(features[labels == 1, 0], features[labels == 1, 1], s=50, c='blue', label='Category 2')
plt.scatter(features[labels == 2, 0], features[labels == 2, 1], s=50, c='green', label='Category 3')
plt.scatter(features[labels == 3, 0], features[labels == 3, 1], s=50, c='orange', label='Category 4')

# 设置中心点的颜色
# center_colors = ['purple', 'cyan', 'black', 'yellow']
center_colors = ['black', 'purple', 'yellow', 'cyan']
for i, color in enumerate(center_colors):
    plt.scatter(centers[i, 0], centers[i, 1], s=200, c=color, marker='X', label=f'Centroid {i+1}')


plt.xlabel('Computational Load')
plt.ylabel('Deadline')
plt.title('Task Clustering')
plt.legend()
plt.grid(True)
plt.show()
