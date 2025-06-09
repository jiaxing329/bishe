from flask import Blueprint, jsonify, request, render_template
from Dao.UserDao import *
import pymysql
from datetime import datetime

# from urllib import request
from flask import Blueprint, jsonify, request

user_api = Blueprint('user_api', __name__)


@user_api.route('/loginApi', methods=['POST'])
def login():
    data = request.get_json()
    print(data)
    username = data.get('username')
    password = data.get('password')
    role = data.get('role')
    res = loginDao(username, password, role)
    print(res[0][5])
    if res.__len__() > 0:
        body = {
            'id': res[0][0],
            'username': res[0][1],
            'password': res[0][2],
            'nickname': res[0][3],
            'sex': res[0][4],
            'age': res[0][5],
            'phone': res[0][6],
            'email': res[0][7],
            'brithday': res[0][8],
            'card': res[0][9],
            'content': res[0][10],
            'remarks': res[0][11],
            'role': res[0][12],
            'token': res[0][0]
        }
        data = {
            'msg': '登录成功',
            'data': body,
            'code': 200
        }
        return jsonify(data)
    else:
        data = {
            'msg': '账号或密码不正确',
            'code': 200
        }
        return jsonify(data)

    return jsonify({
        'msg': '接口报错',
        'code': 500
    })


@user_api.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    res = GetuserDao(username)

    if res.__len__() > 0:
        return jsonify({
            'msg': '该用户名已存在',
            'code': 200
        })
    else:
        Addregister(username, password)
        return jsonify({
            'msg': '注册成功',
            'code': 200
        })


@user_api.route('/userlist', methods=['POST'])
def getuserlist():
    data = request.get_json()
    username = data.get('username')
    page = data.get('page', 1)
    limit = data.get('limit', 20)
    res = ListDao(username=username, page=page, limit=limit)
    data = res['data']
    datalist = []
    for user in data:
        print(user)
        body = {
            'id': user[0],
            'username': user[1],
            'password': user[2],
            'nickname': user[3],
            'sex': user[4],
            'age': user[5],
            'phone': user[6],
            'email': user[7],
            'brithday': user[8],
            'card': user[9],
            'content': user[10],
            'remarks': user[11],
            'role': user[12],
            'token': user[0]
        }
        datalist.append(body)
    data = {
        'msg': '查询成功',
        'data': datalist,
        'code': 200
    }

    return jsonify(data)


@user_api.route('/delete/<int:id>', methods=['POST'])
def delete_user_by_id(id):
    try:
        DeleteUserDao(id)
        return jsonify({
            'msg': '删除成功',
            'code': 200
        })
    except Exception as e:
        return jsonify({
            'msg': '删除失败',
            'code': 500
        })


@user_api.route('/add', methods=['POST'])
def add_user():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        nickname = data.get('nickname')
        sex = data.get('sex')
        age = data.get('age')
        phone = data.get('phone')
        email = data.get('email')

        # 检查用户名是否已存在
        if GetuserDao(username):
            return jsonify({'code': 400, 'msg': '用户名已存在'})

        success = AddUserDao(username, password, nickname, sex, age, phone, email)
        if success:
            return jsonify({'code': 200, 'msg': 'success'})
        else:
            return jsonify({'code': 500, 'msg': '添加失败'})
    except Exception as e:
        return jsonify({'code': 500, 'msg': str(e)})


@user_api.route('/edituser')
def edit_user_page():
    user_id = request.args.get('id')
    user_tuple = get_user_by_id(user_id)
    # 将元组转换为字典
    user = {
        'id': user_tuple[0],
        'username': user_tuple[1],
        'password': user_tuple[2],
        'nickname': user_tuple[3],
        'sex': user_tuple[4],
        'age': user_tuple[5],
        'phone': user_tuple[6],
        'email': user_tuple[7]
    }
    print("用户数据:", user)  # 添加调试输出
    return render_template('edituser.html', user=user)


@user_api.route('/edit', methods=['POST'])
def edit_user():
    data = request.get_json()
    try:
        user_id = data.get('id')
        username = data.get('username')
        password = data.get('password')
        nickname = data.get('nickname')
        sex = data.get('sex')
        age = data.get('age')
        phone = data.get('phone')
        email = data.get('email')

        # 更新用户信息
        success = update_user(user_id, username, password, nickname, sex, age, phone, email)

        if success:
            return jsonify({"code": 200, "message": "更新成功"})
        else:
            return jsonify({"code": 500, "message": "更新失败"})
    except Exception as e:
        return jsonify({"code": 500, "message": str(e)})


@user_api.route('/adminlist', methods=['POST'])
def getadminlist():
    data = request.get_json()
    username = data.get('username')
    page = data.get('page', 1)
    limit = data.get('limit', 20)
    res = ListAdminDao(username=username, page=page, limit=limit)
    data = res['data']
    datalist = []
    for user in data:
        print(user)
        body = {
            'id': user[0],
            'username': user[1],
            'password': user[2],
            'nickname': user[3],
            'sex': user[4],
            'age': user[5],
            'phone': user[6],
            'email': user[7],
            'brithday': user[8],
            'card': user[9],
            'content': user[10],
            'remarks': user[11],
            'role': user[12],
            'token': user[0]
        }
        datalist.append(body)
    data = {
        'msg': '查询成功',
        'data': datalist,
        'code': 200
    }

    return jsonify(data)


@user_api.route('/addadmin', methods=['POST'])
def add_admin():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        nickname = data.get('nickname')
        sex = data.get('sex')
        age = data.get('age')
        phone = data.get('phone')
        email = data.get('email')
        birthday = data.get('birthday')
        card = data.get('card')
        content = data.get('content')
        remarks = data.get('remarks')

        AddAdminDao(username, password, nickname, sex, age, phone, email, birthday, card, content, remarks)
        return jsonify({
            'msg': '添加成功',
            'code': 200
        })
    except Exception as e:
        return jsonify({
            'msg': '添加失败',
            'code': 500
        })


# 添加获取个人信息的路由
@user_api.route('/profile', methods=['GET'])
def get_profile():
    try:
        user_id = request.args.get('id')
        user = get_user_by_id(user_id)
        if user:
            return jsonify({
                'code': 200,
                'msg': '获取成功',
                'data': {
                    'id': user[0],
                    'username': user[1],
                    'nickname': user[3],
                    'sex': user[4],
                    'age': user[5],
                    'phone': user[6],
                    'email': user[7],
                    'birthday': user[8] if len(user) > 8 else None,
                    'card': user[9] if len(user) > 9 else None,
                    'content': user[10] if len(user) > 10 else None,
                    'remarks': user[11] if len(user) > 11 else None
                }
            })
        else:
            return jsonify({
                'code': 404,
                'msg': '用户不存在'
            })
    except Exception as e:
        return jsonify({
            'code': 500,
            'msg': str(e)
        })


# 添加更新个人信息的路由
@user_api.route('/updateProfile', methods=['POST'])
def update_profile():
    try:
        data = request.get_json()
        user_id = data.get('id')

        # 更新用户信息
        success = update_user_profile(
            user_id=user_id,
            nickname=data.get('nickname'),
            sex=data.get('sex'),
            age=data.get('age'),
            phone=data.get('phone'),
            email=data.get('email'),
            birthday=data.get('birthday'),
            card=data.get('card'),
            content=data.get('content'),
            remarks=data.get('remarks')
        )

        if success:
            return jsonify({
                'code': 200,
                'msg': '更新成功'
            })
        else:
            return jsonify({
                'code': 500,
                'msg': '更新失败'
            })
    except Exception as e:
        return jsonify({
            'code': 500,
            'msg': str(e)
        })
