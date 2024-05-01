
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pymysql
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def select_data():
    # 打开数据库连接
    db = pymysql.connect(host="localhost",
                         port=3306,
                         user="root",
                         password="123456",
                         database="mec",
                         charset='utf8')

    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    # sql 语句
    sql = "SELECT * FROM task4"

    try:
        # 使用execute方法执行SQL语句
        cursor.execute(sql)
        # 获取所有结果记录列表
        results = cursor.fetchall()
        # for item in results:
        #     print(item)
    except:
        print("Select Error!")

    # 关闭数据库连接
    db.close()

    return results

def update_data(id, label):
    # 打开数据库连接
    db = pymysql.connect(host="localhost",
                         port=3306,
                         user="root",
                         password="123456",
                         database="mec",
                         charset='utf8')

    # 使用cursor()方法获取操作游标
    cursor = db.cursor()

    # 修改语句
    # sql = "UPDATE task SET cluster_class = " +label + " WHERE task_id = " +id
    sql = "UPDATE task4 SET cluster_class=%s WHERE task_id=%s"
    # print(sql)

    try:
        # 执行
        cursor.execute(sql, (label, id))
        # 提交到数据库执行
        db.commit()
    except:
        # 回滚
        db.rollback()
        print("UPDATE ERROR!")

    # 关闭数据库连接
    db.close()

def cluster():
    results = select_data()
    # 任务
    tasks = np.array(results)
    # 计算任务的计算量（任务大小 * CPU计算密度）
    computational_load = tasks[:, 1] * tasks[:, 3]
    # 计算截止时间的倒数
    # deadline_scaled = 1.0 / tasks[:, 4]
    deadline_scaled = tasks[:, 4]
    # 选择计算量和截止时间作为任务数据
    # tasks_with_computational_load = np.column_stack((computational_load, tasks[:, 4]))
    # 将计算量和截止时间的倒数作为任务数据
    tasks_data = np.column_stack((computational_load, deadline_scaled))
    # tasks_data = np.column_stack((computational_load * deadline_scaled))

    # 数据归一化
    # scaler = StandardScaler()
    # tasks_scaled = scaler.fit_transform(tasks_data)
    # 归一化到 [0, 1]
    scaler = MinMaxScaler()
    tasks_scaled = scaler.fit_transform(tasks_data)

    # 计算中心点的位置
    computational_load_max = np.max(computational_load)
    deadline_max = np.max(deadline_scaled)
    computational_load_min = np.min(computational_load)
    deadline_min = np.min(deadline_scaled)
    computational_load_differ = computational_load_max - computational_load_min
    deadline_differ = deadline_max - deadline_min

    center1 = np.array([1 / 4 * computational_load_differ, 1 / 4 * deadline_differ])
    center2 = np.array([3 / 4 * computational_load_differ, 3 / 4 * deadline_differ])
    center3 = np.array([1 / 4 * computational_load_differ, 3 / 4 * deadline_differ])
    center4 = np.array([3 / 4 * computational_load_differ, 1 / 4 * deadline_differ])
    # 设定四个中心点的位置
    centers = np.vstack((center1, center2, center3, center4))

    # 使用K均值聚类算法
    kmeans = KMeans(n_clusters=4)
    # kmeans = KMeans(n_clusters=4, init=centers, n_init=1, random_state=0)
    # TODO: 使用  tasks_data or tasks_scaled
    kmeans.fit(tasks_data)
    labels = kmeans.labels_

    # 将任务编号和类别进行对应
    # task_categories = np.column_stack((np.arange(len(tasks)), labels + 1))

    # 可视化聚类结果
    # plt.figure(figsize=(8, 6))

    # 可视化聚类结果
    # plt.scatter(tasks_scaled[:, 0], tasks_scaled[:, 1], c=labels, cmap='viridis')
    for i in range(4):
        plt.scatter(computational_load[labels == i], tasks[:, 4][labels == i], label=f'Class {i}')

    plt.xlabel('Computational Load')
    # plt.ylabel('Deadline_inverse')
    plt.ylabel('Deadline')
    plt.title('Task Clustering')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

    return labels

# 输出每个任务的聚类结果
if __name__ == '__main__':
    # 聚类标签
    cluster_labels = cluster()
    for i, label in enumerate(cluster_labels):
        print("任务 {} 属于聚类 {}".format(i, label))
        update_data(i, label)

