# -*- coding: utf-8 -*-

# import MySQLdb
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

'''
try:
    # 使用execute方法执行SQL语句
    cursor.execute(sql)
    # 获取所有结果记录列表
    results = cursor.fetchall()
    # 打印
    for row in results:
        print(row[0] + "," + row[1])
except:
    print("Error!")
'''

try:
    # 使用execute方法执行SQL语句
    cursor.execute(sql)
    # 获取所有结果记录列表
    results = cursor.fetchall()
    print(results)
    # 打印
    # for row in results:
    #     print(row)
except:
    print("Error!")

# 关闭数据库连接
db.close()
