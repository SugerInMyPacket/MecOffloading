import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# 生成任务数据
tasks = np.array([
    [10, 5, 1, 5, 2],    # 任务1：大小=10，CPU计算密度=5，类别=1，截止时间=5，优先级=2
    [20, 3, 2, 7, 1],    # 任务2：大小=20，CPU计算密度=3，类别=2，截止时间=7，优先级=1
    [5, 10, 1, 3, 3],    # 任务3：大小=5，CPU计算密度=10，类别=1，截止时间=3，优先级=3
    [15, 8, 2, 6, 2],    # 任务4：大小=15，CPU计算密度=8，类别=2，截止时间=6，优先级=2
    [25, 4, 1, 8, 1]     # 任务5：大小=25，CPU计算密度=4，类别=1，截止时间=8，优先级=1
])

# 计算任务的计算量和截止时间
computational_load = tasks[:, 0] * tasks[:, 1]
deadline = tasks[:, 3]

# 选择计算量和截止时间作为特征
features = np.column_stack((computational_load, deadline))

# 使用层次聚类算法
Z = linkage(features, 'ward')

# 绘制树状图
plt.figure(figsize=(10, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Tasks')
plt.ylabel('Distance')
dendrogram(Z, labels=np.arange(1, len(tasks) + 1))
plt.show()
