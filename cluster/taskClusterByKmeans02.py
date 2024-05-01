
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
    sql = "UPDATE task SET cluster_class=%s WHERE task_id=%s"
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
    deadline_scaled = 1.0 / tasks[:, 4]
    # deadline_scaled = tasks[:, 4]
    computation_deadline_ratio = computational_load * deadline_scaled

    # 归一化到 [0, 1]
    # scaler = MinMaxScaler()
    # tasks_scaled = scaler.fit_transform(computation_deadline_ratio)

    # 使用K均值聚类算法进行任务分类
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(computation_deadline_ratio.reshape(-1, 1))
    # kmeans.fit(tasks_scaled)

    # 获取聚类结果和聚类中心
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    # 将任务编号和类别进行对应
    task_categories = np.column_stack((np.arange(len(tasks)), cluster_labels + 1))

    # 打印任务编号和对应的类别
    print("任务编号\t类别")
    for task_id, category in task_categories:
        print(f"{int(task_id)}\t\t{int(category)}")

    # 可视化聚类结果
    plt.figure(figsize=(8, 6))

    # 绘制每个类别的任务
    for i in range(4):
        # plt.scatter(computation_deadline_ratio[cluster_labels == i], tasks[:, 3][cluster_labels == i], label=f'Class {i+1}')
        plt.scatter(computational_load[cluster_labels == i], tasks[:, 4][cluster_labels == i], label=f'Class {i+1}')

    # plt.xlabel('Computation/Deadline Ratio')
    plt.xlabel('computational_load')
    plt.ylabel('Deadline')
    plt.title('Task Clustering')
    plt.legend()
    plt.grid(True)
    plt.show()


# 输出每个任务的聚类结果
if __name__ == '__main__':
    # 聚类标签
    cluster_labels = cluster()
    # for i, label in enumerate(cluster_labels):
    #     print("任务 {} 属于聚类 {}".format(i, label))
    #     update_data(i, label)

