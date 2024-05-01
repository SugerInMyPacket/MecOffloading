import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 生成任务数据
tasks = np.array([
    [10, 5, 1, 5, 2],    # 任务1：大小=10，CPU计算密度=5，类别=1，截止时间=5，优先级=2
    [20, 3, 2, 7, 1],    # 任务2：大小=20，CPU计算密度=3，类别=2，截止时间=7，优先级=1
    [5, 10, 1, 3, 3],    # 任务3：大小=5，CPU计算密度=10，类别=1，截止时间=3，优先级=3
    [15, 8, 2, 6, 2],    # 任务4：大小=15，CPU计算密度=8，类别=2，截止时间=6，优先级=2
    [25, 4, 1, 8, 1]     # 任务5：大小=25，CPU计算密度=4，类别=1，截止时间=8，优先级=1
])

# 计算任务的计算量和截止时间的倒数
computational_load = tasks[:, 0] * tasks[:, 1]
deadline_reciprocal = 1 / tasks[:, 3]

# 选择计算量和截止时间的倒数作为特征
features = np.column_stack((computational_load, deadline_reciprocal))

# 使用K均值聚类算法将任务划分为4个类别
kmeans = KMeans(n_clusters=4)
kmeans.fit(features)
labels = kmeans.labels_

# 可视化聚类结果，并按照聚类簇的颜色注明类别
plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis')
plt.xlabel('Computational Load')
plt.ylabel('Deadline Reciprocal')
plt.title('Task Clustering')
cluster_colors = [plt.cm.viridis(i) for i in np.linspace(0, 1, 4)]
plt.colorbar(label='Cluster')

# 添加类别标签和含义说明
legend_labels = {
    0: 'High Computational Load, Small Deadline',
    1: 'High Computational Load, Large Deadline',
    2: 'Low Computational Load, Small Deadline',
    3: 'Low Computational Load, Large Deadline'
}
legend_elements = [Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=cluster_colors[i], markersize=10,
                          label=f'Cluster {i+1}: {legend_labels[i]}') for i in range(4)]
plt.legend(handles=legend_elements, loc='upper right', title='Class Label')

plt.show()
