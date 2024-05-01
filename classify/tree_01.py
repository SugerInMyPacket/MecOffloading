import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. 训练代码并保存模型
# 假设你已经有了训练集数据，命名为train_data.csv
# 加载训练数据
train_data = pd.read_csv('train_data.csv')

# 假设train_data包含了任务属性以及类别信息
X_train = train_data[['任务大小', 'CPU计算密度', '截止时间']]
y_train = train_data['类别']

# 初始化并训练决策树模型
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# 保存模型
joblib.dump(clf, 'decision_tree_model.pkl')

# 2. 测试集上的预测代码
# 假设你有一个名为test_data.csv的测试集文件
# 加载测试数据
test_data = pd.read_csv('test_data.csv')

# 假设测试数据中包含任务属性
X_test = test_data[['任务大小', 'CPU计算密度', '截止时间']]

# 加载保存的模型
saved_model = joblib.load('decision_tree_model.pkl')

# 在测试集上进行预测
predictions = saved_model.predict(X_test)

# 输出预测结果
print("预测结果：", predictions)
