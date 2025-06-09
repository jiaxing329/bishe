from flask import Blueprint, jsonify, request, render_template, current_app, session
from Dao.MusicDao import *
import pymysql
from datetime import datetime
import os
from wordcloud import WordCloud
import jieba
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os.path
import matplotlib.font_manager as fm
from Predictive.RecommendationService import RecommendationService

music_api = Blueprint('music_api', __name__)

# 初始化推荐服务
recommendation_service = RecommendationService()

@music_api.route('/musiclist', methods=['POST'])
def get_music_list():
    try:
        data = request.get_json()
        music_name = data.get('music_name', '')
        page = data.get('page', 1)
        limit = data.get('limit', 20)
        
        res = ListMusicDao(music_name=music_name, page=page, limit=limit)
        data = res['data']
        datalist = []
        
        for music in data:
            body = {
                'id': music[0],
                'music_name': music[1],
                'singer': music[2],
                'languages': music[3],
                'school': music[4],
                'record_company': music[5],
                'release_date': music[6],
                'music_url': music[7],
                'image_url': music[8],
                'play_count': music[9],
                'like_count': music[10],
                'download_count': music[11],
                'comment_count': music[12]
            }
            datalist.append(body)
            
        return jsonify({
            'msg': '查询成功',
            'data': datalist,
            'total': res['total'],
            'code': 200
        })
    except Exception as e:
        return jsonify({
            'msg': f'查询失败: {str(e)}',
            'code': 500
        })

@music_api.route('/add', methods=['POST'])
def add_music():
    try:
        data = request.get_json()
        music_name = data.get('music_name')
        singer = data.get('singer')
        languages = data.get('languages')
        school = data.get('school')
        record_company = data.get('record_company')
        release_date = data.get('release_date')
        music_url = data.get('music_url')
        image_url = data.get('image_url')
        
        success = AddMusicDao(
            music_name, singer, languages, school, 
            record_company, release_date, music_url, image_url
        )
        
        if success:
            return jsonify({
                'msg': '添加成功',
                'code': 200
            })
        else:
            return jsonify({
                'msg': '添加失败',
                'code': 500
            })
    except Exception as e:
        return jsonify({
            'msg': f'添加失败: {str(e)}',
            'code': 500
        })

@music_api.route('/deleteMusic/<int:id>', methods=['POST'])
def delete_music(id):
    try:
        success = DeleteMusicDao(id)
        if success:
            return jsonify({
                'msg': '删除成功',
                'code': 200
            })
        else:
            return jsonify({
                'msg': '删除失败',
                'code': 500
            })
    except Exception as e:
        return jsonify({
            'msg': f'删除失败: {str(e)}',
            'code': 500
        })

@music_api.route('/editmusic')
def edit_music_page():
    music_id = request.args.get('id')
    music_tuple = get_music_by_id(music_id)
    music = {
        'id': music_tuple[0],
        'music_name': music_tuple[1],
        'singer': music_tuple[2],
        'languages': music_tuple[3],
        'school': music_tuple[4],
        'record_company': music_tuple[5],
        'release_date': music_tuple[6],
        'music_url': music_tuple[7],
        'image_url': music_tuple[8]
    }
    return render_template('editmusic.html', music=music)

@music_api.route('/edit', methods=['POST'])
def edit_music():
    try:
        data = request.get_json()
        music_id = data.get('id')
        music_name = data.get('music_name')
        singer = data.get('singer')
        languages = data.get('languages')
        school = data.get('school')
        record_company = data.get('record_company')
        release_date = data.get('release_date')
        music_url = data.get('music_url')
        image_url = data.get('image_url')
        
        success = update_music(
            music_id, music_name, singer, languages, school,
            record_company, release_date, music_url, image_url
        )
        
        if success:
            return jsonify({
                'msg': '更新成功',
                'code': 200
            })
        else:
            return jsonify({
                'msg': '更新失败',
                'code': 500
            })
    except Exception as e:
        return jsonify({
            'msg': f'更新失败: {str(e)}',
            'code': 500
        })

@music_api.route('/play/<int:id>', methods=['POST'])
def play_music(id):
    try:
        success = update_play_count(id)
        if success:
            return jsonify({
                'msg': '播放次数更新成功',
                'code': 200
            })
        else:
            return jsonify({
                'msg': '播放次数更新失败',
                'code': 500
            })
    except Exception as e:
        return jsonify({
            'msg': f'操作失败: {str(e)}',
            'code': 500
        })

@music_api.route('/like/<int:id>', methods=['POST'])
def like_music(id):
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        action = data.get('action', 'like')  # 新增action参数，默认为like
        
        if not user_id:
            return jsonify({
                'msg': '请先登录',
                'code': 401
            })
        
        if action == 'like':
            # 检查是否已点赞
            if check_user_like(id, user_id):
                return jsonify({
                    'msg': '您已经点赞过这首音乐了',
                    'code': 400
                })
            
            # 添加点赞记录
            success = add_like_record(id, user_id)
            msg = '点赞成功' if success else '点赞失败'
        else:
            # 检查是否已点赞
            if not check_user_like(id, user_id):
                return jsonify({
                    'msg': '您还没有点赞过这首音乐',
                    'code': 400
                })
            
            # 取消点赞记录
            success = remove_like_record(id, user_id)
            msg = '取消点赞成功' if success else '取消点赞失败'
        
        if success:
            return jsonify({
                'msg': msg,
                'code': 200
            })
        else:
            return jsonify({
                'msg': msg,
                'code': 500
            })
    except Exception as e:
        return jsonify({
            'msg': f'操作失败: {str(e)}',
            'code': 500
        })

@music_api.route('/download/<int:id>', methods=['POST'])
def download_music(id):
    try:
        success = update_download_count(id)
        if success:
            return jsonify({
                'msg': '下载次数更新成功',
                'code': 200
            })
        else:
            return jsonify({
                'msg': '下载次数更新失败',
                'code': 500
            })
    except Exception as e:
        return jsonify({
            'msg': f'操作失败: {str(e)}',
            'code': 500
        })

@music_api.route('/comments/<int:music_id>', methods=['POST'])
def get_music_comments_api(music_id):
    try:
        data = request.get_json()
        page = data.get('page', 1)
        limit = data.get('limit', 20)
        
        result = get_music_comments(music_id, page, limit)
        
        if result is None:
            return jsonify({
                'msg': '获取评论列表失败',
                'code': 500,
                'total': 0,
                'data': []
            })
            
        return jsonify({
            'msg': '获取成功',
            'code': 200,
            'total': result['total'],
            'data': result['data']
        })
    except Exception as e:
        print(f"获取评论列表失败: {str(e)}")  # 添加错误日志
        return jsonify({
            'msg': f'获取评论列表失败: {str(e)}',
            'code': 500,
            'total': 0,
            'data': []
        })

@music_api.route('/wordcloud/<int:music_id>', methods=['POST'])
def generate_wordcloud(music_id):
    try:
        # 获取音乐的所有评论
        comments = get_all_comments(music_id)
        if not comments:
            return jsonify({
                'msg': '没有找到评论数据',
                'code': 400
            })
        
        # 获取项目根目录
        base_dir = current_app.root_path
        # 输出目录
        output_dir = os.path.join(base_dir, 'static', 'wordcloud')
        
        # 使用项目中的字体文件
        # 尝试多个可能的字体路径
        possible_font_paths = [
            os.path.join(base_dir, 'static', 'fonts', 'SimHei.ttf'),  # static/fonts 目录
            os.path.join(base_dir, 'fonts', 'SimHei.ttf'),           # fonts 目录
            os.path.join(base_dir, '..', 'fonts', 'SimHei.ttf'),     # 项目根目录的 fonts
            os.path.join('fonts', 'SimHei.ttf')                      # 相对于当前目录
        ]
        
        font_path = None
        for path in possible_font_paths:
            if os.path.exists(path):
                font_path = path
                break
        
        # 检查字体文件是否存在
        if font_path is None:
            print("尝试查找字体文件在以下路径:")
            for path in possible_font_paths:
                print(f"- {path} {'(不存在)' if not os.path.exists(path) else '(存在)'}")
            return jsonify({
                'msg': '未找到合适的中文字体，请安装中文字体或运行 python scripts/download_font.py',
                'code': 500
            })
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 合并所有评论文本
        text = ' '.join([comment['comment_content'] for comment in comments])
        
        # 添加结巴分词的词典
        jieba.load_userdict(os.path.join(base_dir, 'static', 'dict', 'music_dict.txt'))
        
        # 使用结巴分词
        words = [word for word in jieba.cut(text) 
                if len(word.strip()) > 1 and not word.isdigit()]  # 过滤掉单字和数字
        
        # 统计词频
        from collections import Counter
        word_freq = Counter(words)
        
        # 过滤掉停用词
        stop_words = {'的', '了', '和', '是', '就', '都', '而', '及', '与', '着',
                     '或', '一个', '没有', '这个', '那个', '这样', '那样', '什么'}
        word_freq = {k: v for k, v in word_freq.items() if k not in stop_words}
        
        try:
            wordcloud = WordCloud(
                font_path=font_path,
                background_color='white',
                width=800,
                height=400,
                max_words=200,
                max_font_size=100,
                prefer_horizontal=0.7,
                collocations=False,
                min_font_size=10,
                random_state=42,
                mode='RGBA'
            ).generate_from_frequencies(word_freq)
            
            # 保存词云图
            image_filename = f'wordcloud_{music_id}.png'
            image_path = os.path.join(output_dir, image_filename)
            
            # 创建图像对象并设置大小
            plt.figure(figsize=(10, 5))
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置显示中文字体
            plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示字符
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            
            # 保存图像并关闭
            plt.savefig(image_path, bbox_inches='tight', pad_inches=0, dpi=300, 
                       facecolor='white', edgecolor='none', transparent=True)
            plt.close()
            
            return jsonify({
                'msg': '生成成功',
                'image_url': f'/static/wordcloud/{image_filename}',
                'code': 200
            })
            
        except FileNotFoundError as e:
            print(f"File not found error: {str(e)}")
            return jsonify({
                'msg': '缺少必要的字体文件，请检查配置',
                'code': 500
            })
            
    except Exception as e:
        print(f"Error details: {str(e)}")
        return jsonify({
            'msg': f'生成词云图失败: {str(e)}',
            'code': 500
        })

@music_api.route('/analysis')
def get_analysis_data():
    try:
        # 获取基础统计数据
        stats = get_music_stats()
        
        # 获取语种分布
        languages = get_languages_distribution()
        
        # 获取流派分布
        schools = get_schools_distribution()
        
        # 获取热门音乐
        top_music = get_top_music()
        
        # 获取发行趋势
        release_trend = get_release_trend()
        
        return jsonify({
            'msg': '获取成功',
            'code': 200,
            'stats': stats,
            'languages': languages,
            'schools': schools,
            'topMusic': top_music,
            'releaseTrend': release_trend
        })
    except Exception as e:
        return jsonify({
            'msg': f'获取数据失败: {str(e)}',
            'code': 500
        })

@music_api.route('/recommendations/<int:user_id>', methods=['GET'])
def get_user_recommendations(user_id):
    try:
        recommendations = get_recommendations(user_id)
        
        if not recommendations:
            return jsonify({
                'code': 200,
                'msg': '获取推荐成功',
                'data': []
            })
            
        return jsonify({
            'code': 200,
            'msg': '获取推荐成功',
            'data': recommendations
        })
    except Exception as e:
        print(f"获取推荐失败: {str(e)}")  # 添加错误日志
        return jsonify({
            'code': 500,
            'msg': f'获取推荐失败: {str(e)}',
            'data': []
        })

@music_api.route('/dashboard/stats')
def get_dashboard_stats():
    try:
        stats = get_music_stats()  # 复用现有的统计方法
        return jsonify({
            'code': 200,
            'stats': stats
        })
    except Exception as e:
        return jsonify({
            'code': 500,
            'msg': f'获取统计数据失败: {str(e)}'
        })

@music_api.route('/dashboard/charts')
def get_dashboard_charts():
    try:
        # 获取发行趋势数据
        release_trend = get_release_trend()
        
        # 获取语种分布
        languages = get_languages_distribution()
        
        # 获取情感分布
        sentiment = get_sentiment_distribution()
        
        # 获取热门评论
        top_comments = get_top_comments()
        
        # 获取时间分布数据
        time_distribution = get_comment_time_distribution()
        
        # 获取关键词数据
        keywords = get_comment_keywords()
        
        return jsonify({
            'code': 200,
            'releaseTrend': release_trend,
            'languages': languages,
            'sentiment': sentiment,
            'topComments': top_comments,
            'timeDistribution': time_distribution,
            'keywords': keywords
        })
    except Exception as e:
        print(f"获取图表数据失败: {str(e)}")  # 添加错误日志
        return jsonify({
            'code': 500,
            'msg': f'获取图表数据失败: {str(e)}'
        })

@music_api.route('/user/likes', methods=['POST'])
def get_user_like_list():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({
                'msg': '请先登录',
                'code': 401
            })
        
        liked_music = get_user_likes(user_id)
        return jsonify({
            'msg': '获取成功',
            'code': 200,
            'data': liked_music
        })
    except Exception as e:
        return jsonify({
            'msg': f'获取失败: {str(e)}',
            'code': 500
        })

@music_api.route('/comment/<int:music_id>', methods=['POST'])
def add_music_comment(music_id):
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        content = data.get('content')
        
        if not user_id:
            return jsonify({
                'msg': '请先登录',
                'code': 401
            })
        
        if not content or len(content.strip()) == 0:
            return jsonify({
                'msg': '评论内容不能为空',
                'code': 400
            })
        
        success = add_comment(music_id, user_id, content.strip())
        if success:
            return jsonify({
                'msg': '评论成功',
                'code': 200
            })
        else:
            return jsonify({
                'msg': '评论失败',
                'code': 500
            })
    except Exception as e:
        return jsonify({
            'msg': f'操作失败: {str(e)}',
            'code': 500
        })

@music_api.route('/update_comments_sentiment', methods=['POST'])
def update_comments_sentiment():
    try:
        success = update_all_comments_sentiment()
        if success:
            return jsonify({
                'code': 200,
                'msg': '评论情感更新成功'
            })
        else:
            return jsonify({
                'code': 500,
                'msg': '评论情感更新失败'
            })
    except Exception as e:
        return jsonify({
            'code': 500,
            'msg': f'操作失败: {str(e)}'
        })

@music_api.route('/delete_comment/<int:comment_id>/<int:music_id>', methods=['POST'])
def delete_music_comment(comment_id, music_id):
    try:
        # 检查用户是否是管理员
        user_info = request.get_json()
        user_role = user_info.get('role')
        
        if user_role != 'admin':
            return jsonify({
                'msg': '只有管理员可以删除评论',
                'code': 403
            })
        
        success = delete_comment(comment_id, music_id)
        if success:
            return jsonify({
                'msg': '删除评论成功',
                'code': 200
            })
        else:
            return jsonify({
                'msg': '删除评论失败',
                'code': 500
            })
    except Exception as e:
        return jsonify({
            'msg': f'删除评论失败: {str(e)}',
            'code': 500
        })
