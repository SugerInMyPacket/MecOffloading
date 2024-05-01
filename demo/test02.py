from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import pymysql

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
sql = "SELECT * FROM task"

try:
    # 使用execute方法执行SQL语句
    cursor.execute(sql)
    # 获取所有结果记录列表
    results = cursor.fetchall()
except:
    print("Error!")

# 关闭数据库连接
db.close()

# 示例任务数据，每个任务有五个属性：任务大小、CPU计算密度、类别、截止时间、优先级
# tasks = np.array([[100, 0.2, 1, 3, 1],
#                   [300, 0.4, 2, 5, 2],
#                   [50, 0.1, 3, 2, 3],
#                   [200, 0.3, 1, 4, 1],
#                   [150, 0.5, 2, 4, 2]])

tasks = np.array(results)
print(tasks)

# 计算任务的计算量（任务大小 * CPU计算密度）
computational_load = tasks[:, 1] * tasks[:, 3]
tasks_with_computational_load = np.column_stack((computational_load, tasks[:, 4]))  # 选择计算量和截止时间作为任务数据

# 数据归一化
scaler = StandardScaler()
tasks_scaled = scaler.fit_transform(tasks_with_computational_load)

# 定义DBSCAN模型并拟合数据
dbscan = DBSCAN(eps=0.5, min_samples=2)
dbscan.fit(tasks_scaled)

# 获取聚类标签
cluster_labels = dbscan.labels_

# 输出每个任务的聚类结果
for i, label in enumerate(cluster_labels):
    print("任务 {} 属于聚类 {}".format(i, label))
