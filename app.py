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

# 基于播放历史的推荐
@app.route('/music/recommendations/history/<int:user_id>')
def get_history_recommendations(user_id):
    try:
        # 获取用户播放历史
        conn = get_connection()
        cursor = conn.cursor()
        
        # 获取用户最近播放的音乐及其特征
        cursor.execute("""
            SELECT 
                m.id,
                m.music_name,
                m.singer,
                m.languages,
                m.school,
                m.image_url,
                m.music_url,
                m.play_count,
                m.like_count,
                m.download_count,
                m.comment_count,
                COUNT(ph.id) as user_play_count,
                MAX(ph.play_time) as last_play_time
            FROM play_history ph
            JOIN music m ON ph.music_id = m.id
            WHERE ph.user_id = %s
            GROUP BY m.id, m.music_name, m.singer, m.languages, m.school, 
                     m.image_url, m.music_url, m.play_count, m.like_count, 
                     m.download_count, m.comment_count
            ORDER BY last_play_time DESC
            LIMIT 50
        """, (user_id,))
        
        history_music = cursor.fetchall()
        
        if not history_music:
            return jsonify({
                'code': 200,
                'msg': 'success',
                'data': []
            })
        
        # 提取用户偏好的音乐特征
        preferred_languages = set()
        preferred_schools = set()
        preferred_singers = set()
        
        for music in history_music:
            if music[3]:  # languages
                preferred_languages.add(music[3])
            if music[4]:  # school
                preferred_schools.add(music[4])
            if music[2]:  # singer
                preferred_singers.add(music[2])
        
        # 获取推荐音乐
        cursor.execute("""
            SELECT 
                id,
                music_name,
                singer,
                languages,
                school,
                image_url,
                music_url,
                play_count,
                like_count,
                download_count,
                comment_count
            FROM music
            WHERE id NOT IN (
                SELECT music_id 
                FROM play_history 
                WHERE user_id = %s
            )
            AND (
                languages IN %s
                OR school IN %s
                OR singer IN %s
            )
            ORDER BY 
                CASE 
                    WHEN languages IN %s THEN 1
                    WHEN school IN %s THEN 2
                    WHEN singer IN %s THEN 3
                    ELSE 4
                END,
                (play_count * 0.4 + like_count * 0.3 + download_count * 0.2 + comment_count * 0.1) DESC
            LIMIT 20
        """, (
            user_id,
            tuple(preferred_languages) or ('',),
            tuple(preferred_schools) or ('',),
            tuple(preferred_singers) or ('',),
            tuple(preferred_languages) or ('',),
            tuple(preferred_schools) or ('',),
            tuple(preferred_singers) or ('',)
        ))
        
        recommended_music = cursor.fetchall()
        
        # 格式化返回数据
        result = []
        for music in recommended_music:
            result.append({
                'id': music[0],
                'music_name': music[1],
                'singer': music[2],
                'languages': music[3],
                'school': music[4],
                'image_url': music[5],
                'music_url': music[6],
                'play_count': music[7],
                'like_count': music[8],
                'download_count': music[9],
                'comment_count': music[10]
            })
        
        return jsonify({
            'code': 200,
            'msg': 'success',
            'data': result
        })
        
    except Exception as e:
        print(f"获取推荐失败: {str(e)}")
        return jsonify({
            'code': 500,
            'msg': f'获取推荐失败: {str(e)}',
            'data': None
        })
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

# 热门推荐
@app.route('/music/recommendations/popular')
def get_popular_recommendations():
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # 获取热门音乐
        cursor.execute("""
            SELECT 
                id,
                music_name,
                singer,
                languages,
                school,
                image_url,
                music_url,
                play_count,
                like_count,
                download_count,
                comment_count
            FROM music
            ORDER BY 
                (play_count * 0.4 + like_count * 0.3 + download_count * 0.2 + comment_count * 0.1) DESC
            LIMIT 20
        """)
        
        popular_music = cursor.fetchall()
        
        # 格式化返回数据
        result = []
        for music in popular_music:
            result.append({
                'id': music[0],
                'music_name': music[1],
                'singer': music[2],
                'languages': music[3],
                'school': music[4],
                'image_url': music[5],
                'music_url': music[6],
                'play_count': music[7],
                'like_count': music[8],
                'download_count': music[9],
                'comment_count': music[10]
            })
        
        return jsonify({
            'code': 200,
            'msg': 'success',
            'data': result
        })
        
    except Exception as e:
        print(f"获取热门推荐失败: {str(e)}")
        return jsonify({
            'code': 500,
            'msg': f'获取热门推荐失败: {str(e)}',
            'data': None
        })
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == '__main__':
    app.run(debug=True)

