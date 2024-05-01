import numpy as np
import pandas as pd
import time

start = time.time()

# 读取数据
read_df = pd.read_csv('../data/task01.csv')

target = read_df.iloc[:, -1]
data = read_df.iloc[:, 1:-1]

k = 7
n = data.shape[0]

# 初始化dis矩阵
dis = np.zeros([n, n])
# 求两两簇（点）之间的距离
for i in range(n - 1):
    for j in range(i + 1, n):
        dis[j][i] = ((data.iloc[j] - data.iloc[i]) ** 2).sum()
    print("初始化dis矩阵进度：{}/{}".format(i + 1, n))
# 下三角复制到上三角
i_lower = np.triu_indices(n, 0)
dis[i_lower] = dis.T[i_lower]
print("初始化dis矩阵进度：{}/{}".format(n, n))


def Exactitude(pre_target, c_num):
    """
    Exactitude的相关定义放在了完整的项目代码中（文末查看）此处不影响使用
    完全预测正确返回0
    """
    pass


######### 以下是重中之重 #########

def regionQuery(p, dis, Eps):
    """
    返回点p的密度直达点
    """
    neighbors = np.where(dis[:, p] <= Eps ** 2)[0]
    return neighbors


def growCluster(dis, pre_target, labels, p, Eps, MinPts):
    """
    寻找p点的所有密度可达点，形成最终一个簇
    输入：距离矩阵、预测标签、初始点p、是否被遍历过的标签、邻域半径、邻域中数据对象数目阈值
    """

    # 如果该点已经经过遍历，结束对该点的操作
    if labels[p] == -1:
        return labels, pre_target

    # p的密度直达点
    NeighborPts = regionQuery(p, dis, Eps)

    # 遍历p的密度直达点
    i = 0
    while i < len(NeighborPts):
        Pn = NeighborPts[i]
        # 找出Pn的密度直达点
        PnNeighborPts = regionQuery(Pn, dis, Eps)
        # 如果此时的点是核心点
        if len(PnNeighborPts) >= MinPts:
            # 将点Pn的新的密度直达点加入点簇
            Setdiff1d = np.setdiff1d(PnNeighborPts, NeighborPts)  # 在PnNeighborPts不在NeighborPts中
            NeighborPts = np.hstack((NeighborPts, Setdiff1d))
        # 否则，说明为边界点，什么也不需要做
        # NeighborPts = NeighborPts
        i += 1

    # 将点p密度可达各点归入p所在簇
    pre_target[NeighborPts] = pre_target[p]
    labels[NeighborPts] = -1
    return labels, pre_target


def DBSCAN(n, k, dis, Eps, MinPts, mode=2):
    """
    输入：距离矩阵、邻域半径、邻域中数据对象数目阈值
    输出：mode==1:预测值准确性（平均标准差），运行时间;mode==2:预测值
    """
    temp_start = int(round(time.time() * 1000000))

    p = 0
    labels = np.zeros(n)  # 有两个可能的值：-1：完成遍历的；0：这个点还没经历过遍历，初始均为0
    pre_target = np.arange(n)

    if mode == 2:
        print("开始循环迭代")

    # 从第一个点开始遍历
    while p < n:
        # 寻找当前点的密度可达点，形成一个簇
        labels, pre_target = growCluster(dis, pre_target, labels, p, Eps, MinPts)
        # 此时的簇数
        c_num = len(np.unique(pre_target))
        if mode == 2:
            print("循环迭代次数：{}，此时有{}个簇".format(p + 1, c_num))
        # 分成小于k簇直接跳出循环（说明分得有问题）
        # 分成正好k簇也跳出循环，直接去检查有没有分对
        if c_num <= k:
            break
        p += 1

    if mode == 2:
        print("结束循环迭代")

    temp_stop = int(round(time.time() * 1000000))

    if mode == 1:
        return Exactitude(pre_target, c_num), temp_stop - temp_start
    elif mode == 2:
        return pre_target


######### 以上是重中之重 #########

# 经过观察，Eps=4.0,MinPts=29可作为参数传入，
# 准确率100%
# 再次提示，测试、参数调整过程及可视化所用相关在文末完整项目中提供
# pre_target = DBSCAN(n=n, k=k, dis=dis, Eps=4.0, MinPts=29, mode=1)

pre_target = DBSCAN(n=n, k=k, dis=dis, Eps=4.0, MinPts=29)

# pca降维
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
newData = pca.fit_transform(data)
newData = pd.DataFrame(newData)

# 可视化
import matplotlib.pyplot as plt

x = np.array(newData.iloc[:, 0])
y = np.array(newData.iloc[:, 1])

# 原数据
plt.subplot(2, 1, 1)
plt.scatter(x, y, c=np.array(target))
# 预测数据
plt.subplot(2, 1, 2)
plt.scatter(x, y, c=pre_target)
plt.show()

end = time.time()
print(end - start)
