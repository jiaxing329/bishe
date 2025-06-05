import pymysql
from datetime import datetime



# 数据库连接
def get_connection():
    return pymysql.connect(
        host='localhost',
        port=3306,
        user='root',
        passwd='123456',
        db='py_music',
        charset='utf8'
    )

# 管理员登陆
def loginDao(username,password,role):
    conn = get_connection()
    cursor3=conn.cursor()
    cursor3.execute('select * from `py_user` where username = %s and password = %s and role = %s' ,(username,password,role))
    res = cursor3.fetchall()
    conn.commit()
    conn.close()
    return res

# 注册检测
def GetuserDao(username):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        sql = "SELECT * FROM py_user WHERE username = %s"
        cursor.execute(sql, (username,))
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

# 用户注册
def Addregister(username, password):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        sql = "INSERT INTO py_user (username, password, role) VALUES (%s, %s, 'user')"
        cursor.execute(sql, (username, password))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"注册失败: {str(e)}")
        return False
    finally:
        cursor.close()
        conn.close()

# 查询用户列表
def ListDao(username='', page=1, limit=20):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # 构建查询条件
        where_clause = "WHERE role = 'user'" # 只查询普通用户
        if username:
            where_clause += f" AND username LIKE '%{username}%'"
        
        # 计算偏移量
        offset = (page - 1) * limit
        
        # 查询总数
        count_sql = f"SELECT COUNT(*) FROM py_user {where_clause}"
        cursor.execute(count_sql)
        total = cursor.fetchone()[0]
        
        # 查询数据
        sql = f"""
            SELECT id, username, password, nickname, sex, age, phone, email, 
                   birthday, card, content, remarks, role 
            FROM py_user 
            {where_clause}
            LIMIT {offset}, {limit}
        """
        cursor.execute(sql)
        data = cursor.fetchall()
        
        return {
            'total': total,
            'data': data
        }
    finally:
        cursor.close()
        conn.close()

# 删除用户
def DeleteUserDao(user_id):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        sql = "DELETE FROM py_user WHERE id = %s AND role = 'user'"
        cursor.execute(sql, (user_id,))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"删除用户失败: {str(e)}")
        return False
    finally:
        cursor.close()
        conn.close()

# 新增用户
def AddUserDao(username, password, nickname, sex, age, phone, email):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        sql = """
            INSERT INTO py_user (username, password, nickname, sex, age, phone, email, role) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'user')
        """
        cursor.execute(sql, (username, password, nickname, sex, age, phone, email))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"添加用户失败: {str(e)}")
        return False
    finally:
        cursor.close()
        conn.close()

# 获取用户详情
def get_user_by_id(user_id):
    """根据ID获取用户信息"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        sql = """
            SELECT id, username, password, nickname, sex, age, phone, email, 
                   birthday, card, content, remarks, role
            FROM py_user 
            WHERE id = %s
        """
        cursor.execute(sql, (user_id,))
        result = cursor.fetchone()
        return result
    except Exception as e:
        print(f"获取用户信息失败: {str(e)}")
        return None
    finally:
        cursor.close()
        conn.close()

# 更新用户信息
def update_user(user_id, username, password, nickname, sex, age, phone, email):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        sql = """
            UPDATE py_user 
            SET username = %s, password = %s, nickname = %s, 
                sex = %s, age = %s, phone = %s, email = %s 
            WHERE id = %s AND role = 'user'
        """
        cursor.execute(sql, (username, password, nickname, sex, age, phone, email, user_id))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"更新用户失败: {str(e)}")
        return False
    finally:
        cursor.close()
        conn.close()

# 查询管理员列表
def ListAdminDao(username=None, page=1, limit=20):
    conn = get_connection()
    cursor4 = conn.cursor()
    
    # 计算偏移量
    offset = (page - 1) * limit
    
    # 构建基础查询语句
    base_sql = "select * from py_user where role ='admin'"
    count_sql = "select count(*) from py_user where role ='admin'"

    # 如果有用户名参数,添加搜索条件
    if username:
        base_sql += " and username like %s"
        count_sql += " and username like %s"
        search_param = f"%{username}%"
        
        # 执行分页查询
        cursor4.execute(base_sql + " LIMIT %s,%s", (search_param, offset, limit))
        # 获取总数
        cursor4.execute(count_sql, (search_param,))
    else:
        # 无搜索条件的查询
        cursor4.execute(base_sql + " LIMIT %s,%s", (offset, limit))
        cursor4.execute(count_sql)
    
    # 获取总记录数
    total = cursor4.fetchone()[0]
    
    # 执行分页查询
    if username:
        cursor4.execute(base_sql + " LIMIT %s,%s", (search_param, offset, limit))
    else:
        cursor4.execute(base_sql + " LIMIT %s,%s", (offset, limit))
    data = cursor4.fetchall()
    
    conn.commit()
    conn.close()
    
    return {
        "total": total,
        "data": data
    }

# 添加管理员
def AddAdminDao(username, password, nickname, sex, age, phone, email, birthday, card, content, remarks):
    conn = get_connection()
    cursor = conn.cursor()
    sql = """insert into py_user(username, password, nickname, sex, age, phone, email, birthday, card, content, remarks, role) 
            values(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'admin')"""
    cursor.execute(sql, (username, password, nickname, sex, age, phone, email, birthday, card, content, remarks))
    conn.commit()
    conn.close()
    return True

def update_user_profile(user_id, nickname, sex, age, phone, email, birthday, card, content, remarks):
    """更新用户个人信息"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # 处理 age 字段
        age = int(age) if age and str(age).strip() else None
        
        sql = """
            UPDATE py_user 
            SET nickname = %s, sex = %s, age = %s, phone = %s, 
                email = %s, birthday = %s, card = %s, content = %s, remarks = %s
            WHERE id = %s
        """
        params = (
            nickname or None,  # 如果为空字符串则设为 None
            sex or None,
            age,
            phone or None,
            email or None,
            birthday or None,
            card or None,
            content or None,
            remarks or None,
            user_id
        )
        cursor.execute(sql, params)
        conn.commit()
        return True
    except Exception as e:
        print(f"更新用户信息失败: {str(e)}")
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()