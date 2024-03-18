import pprint
import pymysql
from pymysql import cursors

#连接mysql数据库
def conn_mysql():
    return pymysql.connect(
        host="localhost",
        port= 3306,
        user= "root",
        password= 'taoqiping',
        database= "code_hunter",
        charset= "utf8"
    )



#查询数据
def query_data(sql):
    conn = conn_mysql()
    try:
        cursor = conn.cursor(pymysql.cursors.DictCursor) #返回数据是字典形式，而不是数组
        cursor.execute(sql)
        return cursor.fetchall()
    finally:
        conn.close()



#更新数据:
def insert_or_update_data(sql):
    conn = conn_mysql()
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit() #提交
    finally:
        conn.close()

'''
    方法作用：插入code_hunter插件使用记录，以便得到更多数据持续训练模型
    @param code_data：代码数据
    @param line_predict：预测出的可能产生异常的代码行
    @param exception_predict：预测出的可能产生的异常类型
    @param user_evaluation：用户评分
    @param line_artificial：人工评价得出的可能产生异常的代码行
    @param exception_artificial：人工评价得出的可能产生的异常类型

'''
def insert_code_hunter_record(code_data, line_predict, exception_predict, user_evaluation, line_artificial, exception_artificial):
    sql = "insert code_hunter_record (code_data,line_predict,exception_predict,user_evaluation,line_artificial,exception_artificial) " \
          "values ('{}','{}','{}',{:d},'{}','{}')".format(code_data, line_predict, exception_predict, user_evaluation, line_artificial, exception_artificial)
    insert_or_update_data(sql)


#尝试执行
if __name__ == "__main__":
    sql = "select * from code_hunter_record"
    datas = query_data(sql)
    pprint.pprint(datas)
