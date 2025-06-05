from flask import Flask, render_template, request, redirect, url_for, jsonify
from Controller.UserApi import *
from Controller.IndexAPI import SystemInfo
from Controller.MusicApi import *
from flask_cors import CORS
import csv
from routes.popular import popular
from Dao.MusicDao import MusicDao
import json

app = Flask(__name__)
CORS(app)
app.secret_key = 'your-secret-key'  # 添加secret key

app.register_blueprint(user_api, url_prefix='/user')
app.register_blueprint(music_api, url_prefix='/music')
app.register_blueprint(popular)

@app.route('/')
def hello_world():  # put application's code here
    return render_template("page-login.html")

@app.route('/Index')
def Home():  # put application's code here
    return render_template("index.html")

@app.route('/register')
def register():
    return render_template("page-register.html")

@app.route('/User')
def User():  # put application's code here
    return render_template("userlist.html")

@app.route('/Admin')
def Admin():  # put application's code here
    return render_template("adminlist.html")

@app.route('/Music')
def Music():  # put application's code here
    return render_template("Musiclist.html")

@app.route('/preferences')
def preferences():
    return render_template("preferences.html")

@app.route('/analysis')
def analysis():
    return render_template("analysis.html")

@app.route('/recommendations')
def recommendations():
    return render_template("recommendations.html")

@app.route('/dashboard')
def dashboard():
    return render_template("dashboard.html")

@app.route('/popular')
def popular():
    return render_template("popular.html")

@app.route('/profile')
def profile():
    return render_template("profile.html")

@app.route('/api/system_info')
def system_info():
    return jsonify(SystemInfo.get_system_info())

@app.route('/api/songs')
def get_songs():
    songs_list = []
    try:
        with open('songs.csv', 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            songs_list = list(reader)
    except Exception as e:
        print(f"读取CSV文件出错: {str(e)}")
    return jsonify(songs_list)

# 添加播放历史
@app.route('/api/add_play_history', methods=['POST'])
def add_play_history_api():
    try:
        data = request.get_json()
        music_id = data.get('music_id')
        user_id = data.get('user_id')  # 从请求体中获取用户ID
        
        if not music_id or not user_id:
            return jsonify({
                'code': 400,
                'msg': '缺少必要参数'
            })
        
        # 添加播放历史记录
        if MusicDao.add_play_history(user_id, music_id):
            return jsonify({
                'code': 200,
                'msg': '添加播放历史成功'
            })
        else:
            return jsonify({
                'code': 500,
                'msg': '添加播放历史失败'
            })
    except Exception as e:
        return jsonify({
            'code': 500,
            'msg': str(e)
        })

# 获取播放历史
@app.route('/api/play_history', methods=['GET'])
def get_play_history_api():
    try:
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 20))
        user_id = request.args.get('user_id')  # 从查询参数中获取用户ID
        
        if not user_id:
            return jsonify({
                'code': 400,
                'msg': '缺少用户ID参数'
            })
        
        # 获取播放历史数据
        result = MusicDao.get_play_history(user_id, page, limit)
        
        return jsonify({
            'code': 200,
            'msg': '获取播放历史成功',
            'data': result
        })
    except Exception as e:
        return jsonify({
            'code': 500,
            'msg': str(e)
        })

# 播放历史页面
@app.route('/history')
def history_page():
    return render_template('history.html')

if __name__ == '__main__':
    app.run(debug=True)

