import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 生成任务数据
tasks = np.array([
    [10, 5, 1, 5, 2],    # 任务1：大小=10，CPU计算密度=5，类别=1，截止时间=5，优先级=2
    [20, 3, 2, 7, 1],    # 任务2：大小=20，CPU计算密度=3，类别=2，截止时间=7，优先级=1
    [5, 10, 1, 3, 3],    # 任务3：大小=5，CPU计算密度=10，类别=1，截止时间=3，优先级=3
    [15, 8, 2, 6, 2],    # 任务4：大小=15，CPU计算密度=8，类别=2，截止时间=6，优先级=2
    [25, 4, 1, 8, 1]     # 任务5：大小=25，CPU计算密度=4，类别=1，截止时间=8，优先级=1
    [10, 5, 1, 5, 2],
    [10, 5, 1, 5, 2],
    [10, 5, 1, 5, 2],
    [10, 5, 1, 5, 2],
    [10, 5, 1, 5, 2],
    [10, 5, 1, 5, 2],
    [10, 5, 1, 5, 2],
    [10, 5, 1, 5, 2],
    [10, 5, 1, 5, 2],
    [10, 5, 1, 5, 2],
    [10, 5, 1, 5, 2],
    [10, 5, 1, 5, 2],
    [10, 5, 1, 5, 2],
    [10, 5, 1, 5, 2]
])

# 计算任务的计算量
computational_load = tasks[:, 0] * tasks[:, 1]

# 选择计算量和截止时间作为特征
features = np.column_stack((computational_load, tasks[:, 3]))

# 使用K均值聚类算法
kmeans = KMeans(n_clusters=2)
kmeans.fit(features)
labels = kmeans.labels_

# 可视化聚类结果
plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis')
plt.xlabel('Computational Load')
plt.ylabel('Deadline')
plt.title('Task Clustering')
plt.colorbar(label='Cluster')
plt.show()
