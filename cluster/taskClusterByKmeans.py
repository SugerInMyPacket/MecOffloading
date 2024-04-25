
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pymysql
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sql_database_name = "task4"
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
    sql = "SELECT * FROM " + sql_database_name

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
    sql = "UPDATE " + sql_database_name + " SET cluster_class=%s WHERE task_id=%s"
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

def cluster(db_name):
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

    # class_0: 计算量 【小】, 截止时间 【短】  --- black
    center1 = np.array([1 / 4 * computational_load_differ, 1 / 4 * deadline_differ])
    # class_1: 计算量 【大】, 截止时间 【短】  --- purple
    center2 = np.array([3 / 4 * computational_load_differ, 1 / 4 * deadline_differ])
    # class_2: 计算量 【小】, 截止时间 【长】  --- yellow
    center3 = np.array([1 / 4 * computational_load_differ, 3 / 4 * deadline_differ])
    # class_3: 计算量 【大】, 截止时间 【长】  --- cyan
    center4 = np.array([3 / 4 * computational_load_differ, 3 / 4 * deadline_differ])
    # 设定四个中心点的位置
    centers = np.vstack((center1, center2, center3, center4))

    # 使用K均值聚类算法
    kmeans = KMeans(n_clusters=4, init=centers, n_init=1, random_state=0)
    # kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
    # n_clusters：表示要聚类的簇的数量，即你希望将数据分成的组数。
    # init：表示初始化中心点的方法。可以是以下几种选择：
    # · 'k-means++'：使用k-means++算法来选择初始的聚类中心，默认值。
    # · 'random'：随机选择数据中的点作为初始聚类中心。
    # · 也可以传入一个ndarray类型的值作为初始的聚类中心。
    # n_init：表示在不同的随机种子下运行k均值算法的次数，然后返回最好的聚类结果。默认值为10。
    # max_iter：表示单次运行中K均值算法的最大迭代次数。默认值为300。
    # random_state：表示随机数种子，用于初始化聚类中心。设置一个随机种子可以保证每次运行算法得到相同的结果。
    # n_jobs：表示并行计算的数量。如果为 - 1，则使用所有CPU进行计算。默认值为None，表示不并行计算。
    # TODO: 使用  tasks_data or tasks_scaled
    kmeans.fit(tasks_data)
    # 获取目标函数值随着迭代次数的变化
    # 用于存储簇内平方和的列表
    # inertia_values = []
    # 迭代训练模型，并记录每次迭代后的簇内平方和
    # for i in range(1, 11):
    #     kmeans.fit(tasks_data)
    #     inertia_values.append(kmeans.inertia_)

    labels = kmeans.labels_


    # 可视化聚类结果
    # plt.figure(figsize=(8, 6))

    # 可视化聚类结果
    # plt.scatter(tasks_scaled[:, 0], tasks_scaled[:, 1], c=labels, cmap='viridis')
    task_class = ["CL_S&DL_S","CL_H&DL_S","CL_S&DL_H","CL_H&DL_H"]
    for i in range(4):
        # plt.scatter(computational_load[labels == i], tasks[:, 4][labels == i], label=f'Class {i}')
        plt.scatter(computational_load[labels == i], tasks[:, 4][labels == i], label=task_class[i])

    # 设置中心点的颜色
    center_colors = ['black', 'purple', 'yellow', 'cyan']
    for i, color in enumerate(center_colors):
        # plt.scatter(centers[i:, 0], centers[i:, 1], s=200, c=color, marker='X', label=f'Centroid {i+1}')
        # plt.scatter(centers[i:, 0], centers[i:, 1], s=200, c=color, marker='X')
        # plt.scatter(kmeans.cluster_centers_[i:, 0], kmeans.cluster_centers_[i:, 1], s=300, c=color, marker='X', label='Centroids')
        plt.scatter(kmeans.cluster_centers_[i:, 0], kmeans.cluster_centers_[i:, 1], s=100, c=color, marker='X')

    plt.xlabel('Computational Load')
    # plt.ylabel('Deadline_inverse')
    plt.ylabel('Deadline')
    # plt.title(db_name + ' Clustering')
    plt.title('Clustering')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

    # 可视化簇内平方和随迭代次数的变化
    # plt.plot(range(1, 11), inertia_values, marker='o')
    # plt.title('Inertia vs Iteration')
    # plt.xlabel('Number of Iterations')
    # plt.ylabel('Inertia')
    # plt.grid(True)
    # plt.show()

    return labels


# 输出每个任务的聚类结果
if __name__ == '__main__':
    sql_database_name = "task_100_v50"
    # 聚类标签
    cluster_labels = cluster(sql_database_name)
    # 遍历
    for i, label in enumerate(cluster_labels):
        print("任务 {} 属于聚类 {}".format(i, label))
        # 修改task类别
        # update_data(i, label)

