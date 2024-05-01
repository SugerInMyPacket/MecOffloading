from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 示例任务数据，每个任务有五个属性：任务大小、CPU计算密度、类别、截止时间、优先级
tasks = np.array([[100, 0.2, 1, 3, 1],
                  [300, 0.4, 2, 5, 2],
                  [50, 0.1, 3, 2, 3],
                  [200, 0.3, 1, 4, 1],
                  [150, 0.5, 2, 4, 2]])

# 计算任务的计算量（任务大小 * CPU计算密度）
computational_load = tasks[:, 0] * tasks[:, 1]
# 计算截止时间的倒数
inverse_deadline = 1 / tasks[:, 3]
# 选择计算量和截止时间作为任务数据
# tasks_with_computational_load = np.column_stack((computational_load, tasks[:, 3]))

# 将计算量和截止时间的倒数作为任务数据
tasks_data = np.column_stack((computational_load, inverse_deadline))

# 数据归一化
scaler = StandardScaler()
tasks_scaled = scaler.fit_transform(tasks_data)

# 定义DBSCAN模型并拟合数据
dbscan = DBSCAN(eps=0.5, min_samples=2)
dbscan.fit(tasks_scaled)

# 获取聚类标签
cluster_labels = dbscan.labels_

# 可视化聚类结果
plt.figure(figsize=(8, 6))

# 绘制各个簇的数据点
unique_labels = set(cluster_labels)
for label in unique_labels:
    if label == -1:
        # 如果是噪声点，单独绘制
        cluster_mask = (cluster_labels == label)
        plt.scatter(tasks_scaled[cluster_mask, 0], tasks_scaled[cluster_mask, 1], label='Noise', color='gray', alpha=0.5)
    else:
        # 绘制簇内的数据点
        cluster_mask = (cluster_labels == label)
        plt.scatter(tasks_scaled[cluster_mask, 0], tasks_scaled[cluster_mask, 1], label='Cluster {}'.format(label))

plt.title('DBSCAN Clustering of Tasks')
plt.xlabel('Scaled Computational Load')
plt.ylabel('Scaled Deadline')
plt.legend()
plt.show()
