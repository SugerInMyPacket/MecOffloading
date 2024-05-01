import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 生成示例数据，假设有100个任务
np.random.seed(0)
n_samples = 100
task_size = np.random.randint(1, 100, size=(n_samples, 1))
cpu_density = np.random.uniform(0.1, 1, size=(n_samples, 1))
deadline = np.random.randint(1, 100, size=(n_samples, 1))

# 计算任务的计算量
computational_load = task_size * cpu_density

# 标准化数据
data = np.concatenate((task_size, cpu_density, computational_load, deadline), axis=1)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# 使用K-means++算法进行聚类
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0)
kmeans.fit(scaled_data)

# 获取聚类结果和聚类中心
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# 可视化聚类结果
plt.figure(figsize=(10, 6))

# 绘制原始数据点
# plt.scatter(data[:, 2], data[:, 3], c=labels, cmap='viridis', s=50, alpha=0.5, label='Tasks')
plt.scatter(scaled_data[:, 2], scaled_data[:, 3], c=labels, cmap='viridis', s=50, alpha=0.5, label='Tasks')

# 绘制聚类中心
plt.scatter(centers[:, 2], centers[:, 3], c='red', marker='x', s=200, label='Cluster Centers')

plt.xlabel('Computational Load')
plt.ylabel('Deadline')
plt.title('Task Clustering')
plt.legend()
plt.grid(True)
plt.show()

# 输出各个类别的任务数量
for i in range(4):
    print(f"Cluster {i+1}: {np.sum(labels==i)} tasks")
