import numpy as np
import matplotlib.pyplot as plt
import pymysql
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


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

def trainData():

    # 训练决策树模型
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # 保存模型到本地
    joblib.dump(clf, 'decision_tree_model.pkl')

    # 可视化决策树
    # plt.figure(figsize=(10, 10))
    # plot_tree(clf, filled=True, feature_names=['Task Size', 'Deadline'], class_names=['Class 1', 'Class 2', 'Class 3', 'Class 4'])
    # plt.show()


    # 记录训练过程中的准确率和深度
    train_accuracies = []
    test_accuracies = []
    depths = []

    # 训练决策树模型并记录训练过程
    for depth in range(1, 6):  # 假设最大深度为5
        clf.set_params(max_depth=depth)
        clf.fit(X_train, y_train)

        # 记录训练和测试集的准确率
        train_predictions = clf.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_predictions)
        train_accuracies.append(train_accuracy)

        test_predictions = clf.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_predictions)
        test_accuracies.append(test_accuracy)

        # 记录当前深度
        depths.append(depth)

        # 可视化决策树
        # from sklearn.tree import plot_tree
        # plt.figure(figsize=(10, 6))
        # plot_tree(clf, filled=True, feature_names=['Task Size', 'Deadline'],
        #           class_names=['Class 1', 'Class 2', 'Class 3', 'Class 4'])
        # plt.title(f'Decision Tree (Max Depth: {depth})')
        # plt.show()

    # 绘制准确率随深度变化的曲线
    plt.plot(depths, train_accuracies, label='Train Accuracy')
    plt.plot(depths, test_accuracies, label='Test Accuracy')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree Accuracy vs Max Depth')
    plt.legend()
    plt.show()



def testData():
    # 使用模型进行预测
    # 假设你已经有了一个包含测试数据的numpy数组，X_test
    # 请替换下面的示例数据为你自己的测试数据
    # 加载模型
    loaded_model = joblib.load('decision_tree_model.pkl')

    # 预测测试集的类别
    predictions = loaded_model.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:", accuracy)

    # 可视化聚类结果
    # 假设任务属性包含两个特征
    # 请替换下面的示例代码为你自己的数据可视化方法
    plt.figure(figsize=(8, 6))

    # 根据类别绘制不同颜色的点
    for i in range(1, 5):
        plt.scatter(X[y == i, 0], X[y == i, 1], label=f'Class {i}')

    plt.title('Task Clustering')
    plt.xlabel('Computation Load')
    plt.ylabel('Deadline')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    sql_database_name = "task2"
    # 数据库查询结果
    results = select_data()
    # 任务
    tasks = np.array(results)
    # 计算任务的计算量（任务大小 * CPU计算密度）
    computational_load = tasks[:, 1] * tasks[:, 3]
    # 计算截止时间的倒数
    # deadline_scaled = 1.0 / tasks[:, 4]
    deadline_scaled = tasks[:, 4]

    class_labels = tasks[:, 9]

    # 生成示例数据集
    # 假设你已经有了一个包含任务属性的numpy数组，X，和一个包含任务类别的numpy数组，y
    # 请替换下面的示例数据为你自己的数据
    # X = np.array(computational_load, deadline_scaled)
    X = np.array(list(zip(computational_load, deadline_scaled)))
    y = class_labels

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # 训练
    trainData()
    # 测试数据
    # testData()

