import pymysql

def select_data(db_name):
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
    sql = "SELECT * FROM " + db_name

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

# 修改task计算密度
def update_sql_data_c(db_name, vals):
    # 打开数据库连接
    db = pymysql.connect(host="localhost",
                         port=3306,
                         user="root",
                         password="123456",
                         database="mec",
                         charset='utf8')

    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    # 获取所有行
    cursor.execute("SELECT * FROM" + db_name)
    rows = cursor.fetchall()

    # sql 语句
    sql = "UPDATE " + db_name + " SET c=%s WHERE task_id=%s"

    try:
        # 执行
        for i, row in enumerate(rows):
            value_index = i % len(vals)
            val = vals[value_index]
            cursor.execute(sql, (val, row[0]))
        # 提交到数据库执行
        db.commit()
    except:
        # 回滚
        db.rollback()
        print("UPDATE ERROR!")

    # 关闭数据库连接
    db.close()


if __name__ == '__main__':
    # 修改数据操作
    sql_db_name = " task2 "

    # result = select_data(sql_db_name)
    # 修改计算量 c 为 val
    # vals = ["10", "50", "200"]
    vals = [10, 50, 200]
    update_sql_data_c(sql_db_name, vals)


