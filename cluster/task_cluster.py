from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import pymysql
import matplotlib.pyplot as plt

'''
# 打开数据库连接
db = pymysql.connect(host="localhost",
                     port=3306,
                     user="root",
                     password="123456",
                     database="mec",
                     charset='utf8')

# 使用cursor()方法获取操作游标
cursor = db.cursor()
'''


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
    except:
        print("Select Error!")

    # 关闭数据库连接
    db.close()

    return results


def cluster():
    results = select_data()
    # 任务
    tasks = np.array(results)
    # print(tasks)

    # 计算任务的计算量（任务大小 * CPU计算密度）
    computational_load = tasks[:, 1] * tasks[:, 3]
    # 计算截止时间的倒数
    inverse_deadline = 1.0 / tasks[:, 4]
    # inverse_deadline = tasks[:, 4]
    # 选择计算量和截止时间作为任务数据
    # tasks_with_computational_load = np.column_stack((computational_load, tasks[:, 4]))
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

    # 绘制各个簇的数据点
    unique_labels = set(cluster_labels)
    for label in unique_labels:
        if label == -1:
            # 如果是噪声点，单独绘制
            cluster_mask = (cluster_labels == label)
            plt.scatter(tasks_scaled[cluster_mask, 0], tasks_scaled[cluster_mask, 1], label='Noise', color='gray',
                        alpha=0.5)
        else:
            # 绘制簇内的数据点
            cluster_mask = (cluster_labels == label)
            plt.scatter(tasks_scaled[cluster_mask, 0], tasks_scaled[cluster_mask, 1], label='Cluster {}'.format(label))

    plt.title('DBSCAN Clustering of Tasks')
    plt.xlabel('Scaled Computational Load')
    plt.ylabel('Scaled Deadline')
    plt.legend()
    plt.show()

    return cluster_labels


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


# 输出每个任务的聚类结果
if __name__ == '__main__':
    # 聚类标签
    cluster_labels = cluster()

    for i, label in enumerate(cluster_labels):
        print("任务 {} 属于聚类 {}".format(i, label))
        update_data(i, label)
