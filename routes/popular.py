from flask import Blueprint, render_template, request, jsonify, make_response
from Predictive.RecommendationService import RecommendationService

popular = Blueprint('popular', __name__)
recommendation_service = RecommendationService()

@popular.route('/popular')
def popular_page():
    # 获取分类信息
    categories = recommendation_service.get_music_categories()
    
    # 获取默认的流行音乐列表（总榜）
    popular_music = recommendation_service.get_popular_music_by_type()
    
    response = make_response(render_template('popular.html',
                                           categories=categories,
                                           popular_music=popular_music))
    # 禁用缓存
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@popular.route('/api/popular')
def get_popular():
    type_filter = request.args.get('type')  # 如 'languages:华语'
    time_range = request.args.get('time', 'all')  # week/month/year/all
    limit = int(request.args.get('limit', 12))
    
    popular_music = recommendation_service.get_popular_music_by_type(
        type_filter=type_filter,
        time_range=time_range,
        limit=limit
    )
    
    return jsonify(popular_music) 