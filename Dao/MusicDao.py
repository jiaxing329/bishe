import pymysql
from datetime import datetime
from utils.SentimentAnalyzer import SentimentAnalyzer

# 数据库连接
def get_connection():
    return pymysql.connect(
        host='localhost',
        port=3306,
        user='root',
        passwd='123456',
        db='py_music',
        charset='utf8mb4'
    )

# 查询音乐列表
def ListMusicDao(music_name='', page=1, limit=20):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # 构建查询条件
        where_clause = "WHERE 1=1"
        if music_name:
            where_clause += f" AND music_name LIKE '%{music_name}%'"
        
        # 计算偏移量
        offset = (page - 1) * limit
        
        # 查询总数
        count_sql = f"SELECT COUNT(*) FROM music {where_clause}"
        cursor.execute(count_sql)
        total = cursor.fetchone()[0]
        
        # 查询数据
        sql = f"""
            SELECT id, music_name, singer, languages, school, record_company, 
                   release_date, music_url, image_url, play_count, like_count, 
                   download_count, comment_count
            FROM music 
            {where_clause}
            ORDER BY id DESC
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

# 添加音乐
def AddMusicDao(music_name, singer, languages, school, record_company, 
                release_date, music_url, image_url):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        sql = """
            INSERT INTO music (music_name, singer, languages, school, record_company, 
                             release_date, music_url, image_url, play_count, 
                             like_count, download_count, comment_count) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 0, 0, 0, 0)
        """
        cursor.execute(sql, (music_name, singer, languages, school, record_company,
                           release_date, music_url, image_url))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"添加音乐失败: {str(e)}")
        return False
    finally:
        cursor.close()
        conn.close()

# 删除音乐
def DeleteMusicDao(music_id):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # 首先删除相关的评论
        cursor.execute("DELETE FROM music_comments WHERE music_id = %s", (music_id,))
        # 然后删除音乐
        cursor.execute("DELETE FROM music WHERE id = %s", (music_id,))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"删除音乐失败: {str(e)}")
        return False
    finally:
        cursor.close()
        conn.close()

# 获取音乐详情
def get_music_by_id(music_id):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        sql = """
            SELECT id, music_name, singer, languages, school, record_company, 
                   release_date, music_url, image_url, play_count, like_count, 
                   download_count, comment_count
            FROM music 
            WHERE id = %s
        """
        cursor.execute(sql, (music_id,))
        return cursor.fetchone()
    finally:
        cursor.close()
        conn.close()

# 更新音乐信息
def update_music(music_id, music_name, singer, languages, school, record_company, 
                release_date, music_url, image_url):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        sql = """
            UPDATE music 
            SET music_name = %s, singer = %s, languages = %s, school = %s, 
                record_company = %s, release_date = %s, music_url = %s, image_url = %s
            WHERE id = %s
        """
        cursor.execute(sql, (music_name, singer, languages, school, record_company,
                           release_date, music_url, image_url, music_id))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"更新音乐失败: {str(e)}")
        return False
    finally:
        cursor.close()
        conn.close()

# 更新播放次数
def update_play_count(music_id):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        sql = "UPDATE music SET play_count = play_count + 1 WHERE id = %s"
        cursor.execute(sql, (music_id,))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"更新播放次数失败: {str(e)}")
        return False
    finally:
        cursor.close()
        conn.close()

# 更新点赞次数
def update_like_count(music_id):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        sql = "UPDATE music SET like_count = like_count + 1 WHERE id = %s"
        cursor.execute(sql, (music_id,))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"更新点赞次数失败: {str(e)}")
        return False
    finally:
        cursor.close()
        conn.close()

# 更新下载次数
def update_download_count(music_id):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        sql = "UPDATE music SET download_count = download_count + 1 WHERE id = %s"
        cursor.execute(sql, (music_id,))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"更新下载次数失败: {str(e)}")
        return False
    finally:
        cursor.close()
        conn.close()

# 获取音乐评论
def get_music_comments(music_id, page=1, limit=20):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # 计算偏移量
        offset = (page - 1) * limit
        
        # 查询总数
        count_sql = "SELECT COUNT(*) FROM music_comments WHERE music_id = %s"
        cursor.execute(count_sql, (music_id,))
        total = cursor.fetchone()[0]
        
        # 查询评论数据，添加评论ID
        sql = """
            SELECT 
                mc.id,
                mc.comment_content,
                mc.comment_time,
                mc.like_count,
                mc.sentiment
            FROM music_comments mc
            WHERE mc.music_id = %s
            ORDER BY mc.comment_time DESC
            LIMIT %s, %s
        """
        cursor.execute(sql, (music_id, offset, limit))
        data = cursor.fetchall()
        
        # 格式化返回数据
        comments = []
        for row in data:
            comments.append({
                'id': row[0],
                'comment_content': row[1] if row[1] else '',
                'comment_time': row[2].strftime('%Y-%m-%d %H:%M:%S') if row[2] else '',
                'like_count': row[3] if row[3] is not None else 0,
                'sentiment': row[4] if row[4] else '中性'
            })
        
        return {
            'total': total,
            'data': comments
        }
    except Exception as e:
        print(f"获取评论列表失败: {str(e)}")
        return {
            'total': 0,
            'data': []
        }
    finally:
        cursor.close()
        conn.close()

# 获取音乐的所有评论
def get_all_comments(music_id):
    """获取音乐的所有评论"""
    conn = get_connection()
    cursor = conn.cursor()  # 使用普通游标
    try:
        sql = """
            SELECT comment_content
            FROM music_comments 
            WHERE music_id = %s
        """
        cursor.execute(sql, (music_id,))
        comments = cursor.fetchall()
        # 将元组转换为字典格式
        return [{'comment_content': comment[0]} for comment in comments]
    finally:
        cursor.close()
        conn.close()

def get_music_stats():
    """获取音乐基础统计数据"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # 获取音乐总数
        cursor.execute("SELECT COUNT(*) FROM music")
        total_music = cursor.fetchone()[0]
        
        # 获取总播放量
        cursor.execute("SELECT SUM(play_count) FROM music")
        total_plays = cursor.fetchone()[0] or 0
        
        # 获取总点赞数
        cursor.execute("SELECT SUM(like_count) FROM music")
        total_likes = cursor.fetchone()[0] or 0
        
        # 获取总评论数
        cursor.execute("SELECT SUM(comment_count) FROM music")
        total_comments = cursor.fetchone()[0] or 0
        
        return {
            'totalMusic': total_music,
            'totalPlays': total_plays,
            'totalLikes': total_likes,
            'totalComments': total_comments
        }
    finally:
        cursor.close()
        conn.close()

def get_languages_distribution():
    """获取语种分布"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT languages, COUNT(*) as count 
            FROM music 
            WHERE languages IS NOT NULL AND languages != ''
            GROUP BY languages 
            ORDER BY count DESC
        """)
        return [{'language': row[0], 'count': row[1]} for row in cursor.fetchall()]
    finally:
        cursor.close()
        conn.close()

def get_schools_distribution():
    """获取流派分布"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT school, COUNT(*) as count 
            FROM music 
            WHERE school IS NOT NULL AND school != ''
            GROUP BY school 
            ORDER BY count DESC
        """)
        return [{'school': row[0], 'count': row[1]} for row in cursor.fetchall()]
    finally:
        cursor.close()
        conn.close()

def get_top_music():
    """获取热门音乐 TOP 10"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT music_name, play_count 
            FROM music 
            ORDER BY play_count DESC 
            LIMIT 10
        """)
        return [{'music_name': row[0], 'play_count': row[1]} for row in cursor.fetchall()]
    finally:
        cursor.close()
        conn.close()

def get_release_trend():
    """获取发行趋势（最近12个月）"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT DATE_FORMAT(release_date, '%Y-%m') as month, 
                   COUNT(*) as count
            FROM music 
            WHERE release_date IS NOT NULL
            GROUP BY month 
            ORDER BY month DESC 
            LIMIT 12
        """)
        return [{'month': row[0], 'count': row[1]} for row in cursor.fetchall()]
    finally:
        cursor.close()
        conn.close()

def get_user_activity():
    """获取用户活跃度数据（最近30天）"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT 
                DATE(NOW()) - INTERVAL (a.a + (10 * b.a) + (100 * c.a)) DAY as date,
                COUNT(mc.id) as comment_count,
                SUM(mc.like_count) as like_count
            FROM 
                (SELECT 0 as a UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) as a
                CROSS JOIN
                (SELECT 0 as a UNION ALL SELECT 1 UNION ALL SELECT 2) as b
                CROSS JOIN
                (SELECT 0 as a) as c
                LEFT JOIN music_comments mc ON DATE(NOW()) - INTERVAL (a.a + (10 * b.a) + (100 * c.a)) DAY = DATE(mc.comment_time)
            WHERE 
                DATE(NOW()) - INTERVAL (a.a + (10 * b.a) + (100 * c.a)) DAY >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
            GROUP BY date
            ORDER BY date
        """)
        return [
            {
                'date': row[0].strftime('%Y-%m-%d'),
                'comment_count': row[1] or 0,
                'like_count': row[2] or 0
            }
            for row in cursor.fetchall()
        ]
    finally:
        cursor.close()
        conn.close()

def add_comment(music_id, user_id, content):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # 分析评论情感
        sentiment = SentimentAnalyzer.analyze(content)
        
        # 开始事务
        conn.begin()
        
        # 添加评论记录
        sql1 = """
            INSERT INTO music_comments 
            (music_id, user_id, comment_content, comment_time, like_count, sentiment) 
            VALUES (%s, %s, %s, NOW(), 0, %s)
        """
        cursor.execute(sql1, (music_id, user_id, content, sentiment))
        
        # 更新音乐评论数
        sql2 = "UPDATE music SET comment_count = comment_count + 1 WHERE id = %s"
        cursor.execute(sql2, (music_id,))
        
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"添加评论失败: {str(e)}")
        return False
    finally:
        cursor.close()
        conn.close()

def get_sentiment_distribution():
    """获取评论情感分布"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # 修改基础结果集，只包含积极和消极
        base_sentiments = [
            {'name': '积极', 'value': 0},
            {'name': '消极', 'value': 0}
        ]
        
        # 修改查询，只统计非空的情感值
        cursor.execute("""
            SELECT 
                sentiment,
                COUNT(*) as count
            FROM music_comments
            WHERE sentiment IS NOT NULL
            GROUP BY sentiment
        """)
        
        # 获取查询结果
        results = cursor.fetchall()
        
        # 更新基础结果集中的数值
        sentiment_counts = {row[0]: row[1] for row in results}
        for item in base_sentiments:
            item['value'] = sentiment_counts.get(item['name'], 0)
        
        return base_sentiments
        
    finally:
        cursor.close()
        conn.close()

def get_comment_time_distribution():
    """获取24小时评论时间分布"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT 
                HOUR(comment_time) as hour,
                COUNT(*) as count
            FROM music_comments
            GROUP BY HOUR(comment_time)
            ORDER BY hour
        """)
        
        # 创建24小时的基础数据
        hours_data = {hour: 0 for hour in range(24)}
        
        # 更新实际数据
        for row in cursor.fetchall():
            hour = row[0]
            if hour in hours_data:
                hours_data[hour] = row[1]
        
        # 转换为列表格式
        return [{'hour': hour, 'count': count} 
                for hour, count in hours_data.items()]
    finally:
        cursor.close()
        conn.close()

def get_top_comments():
    """获取热门评论 TOP 10"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT 
                mc.comment_content,
                mc.like_count,
                COUNT(mr.id) as reply_count,
                m.music_name,
                (mc.like_count + COUNT(mr.id) * 2) as weight
            FROM music_comments mc
            LEFT JOIN music_reply mr ON mc.id = mr.comment_id
            JOIN music m ON mc.music_id = m.id
            GROUP BY mc.id, mc.comment_content, mc.like_count, m.music_name
            ORDER BY weight DESC
            LIMIT 10
        """)
        
        return [{
            'content': row[0],
            'likes': row[1],
            'replies': row[2],
            'music_name': row[3],
            'weight': row[4]
        } for row in cursor.fetchall()]
    finally:
        cursor.close()
        conn.close()

def get_comment_keywords():
    """获取评论关键词及其情感色彩"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT 
                keyword,
                sentiment,
                COUNT(*) as count
            FROM comment_keywords
            GROUP BY keyword, sentiment
            HAVING count >= 5
            ORDER BY count DESC
            LIMIT 100
        """)
        
        return [{
            'keyword': row[0],
            'sentiment': row[1] or '中性',
            'count': row[2]
        } for row in cursor.fetchall()]
    finally:
        cursor.close()
        conn.close()

# 检查用户是否已点赞
def check_user_like(music_id, user_id):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        sql = "SELECT id FROM music_likes WHERE music_id = %s AND user_id = %s"
        cursor.execute(sql, (music_id, user_id))
        result = cursor.fetchone()
        return result is not None
    finally:
        cursor.close()
        conn.close()

# 添加点赞记录
def add_like_record(music_id, user_id):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # 检查是否已经点赞
        if check_user_like(music_id, user_id):
            return False
            
        # 开始事务
        conn.begin()
        
        # 添加点赞记录
        sql1 = "INSERT INTO music_likes (music_id, user_id) VALUES (%s, %s)"
        cursor.execute(sql1, (music_id, user_id))
        
        # 更新音乐点赞数
        sql2 = "UPDATE music SET like_count = like_count + 1 WHERE id = %s"
        cursor.execute(sql2, (music_id,))
        
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"添加点赞记录失败: {str(e)}")
        return False
    finally:
        cursor.close()
        conn.close()

# 获取用户点赞列表
def get_user_likes(user_id):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        sql = "SELECT music_id FROM music_likes WHERE user_id = %s"
        cursor.execute(sql, (user_id,))
        return [row[0] for row in cursor.fetchall()]
    finally:
        cursor.close()
        conn.close()

# 取消点赞记录
def remove_like_record(music_id, user_id):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # 开始事务
        conn.begin()
        
        # 删除点赞记录
        sql1 = "DELETE FROM music_likes WHERE music_id = %s AND user_id = %s"
        cursor.execute(sql1, (music_id, user_id))
        
        # 更新音乐点赞数
        sql2 = "UPDATE music SET like_count = like_count - 1 WHERE id = %s"
        cursor.execute(sql2, (music_id,))
        
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"取消点赞记录失败: {str(e)}")
        return False
    finally:
        cursor.close()
        conn.close()

def update_all_comments_sentiment():
    """重新分析所有评论的情感"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # 开始事务
        conn.begin()
        
        # 获取所有评论
        cursor.execute("SELECT id, comment_content FROM music_comments")
        comments = cursor.fetchall()
        
        # 更新计数器
        updated_count = 0
        total_count = len(comments)
        
        # 逐个更新评论的情感
        for comment_id, content in comments:
            if content:
                # 分析情感
                sentiment = SentimentAnalyzer.analyze(content)
                
                # 更新数据库
                update_sql = "UPDATE music_comments SET sentiment = %s WHERE id = %s"
                cursor.execute(update_sql, (sentiment, comment_id))
                updated_count += 1
                
                # 每100条提交一次，避免事务太大
                if updated_count % 100 == 0:
                    conn.commit()
                    print(f"已更新 {updated_count}/{total_count} 条评论")
        
        # 提交剩余的更新
        conn.commit()
        print(f"评论情感更新完成，共更新 {updated_count} 条评论")
        return True
    except Exception as e:
        conn.rollback()
        print(f"更新评论情感失败: {str(e)}")
        return False
    finally:
        cursor.close()
        conn.close()
# update_all_comments_sentiment()

def get_recommendations(user_id):
    """获取用户的音乐推荐"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # 1. 获取用户的音乐偏好
        preference_sql = """
            SELECT 
                m.languages,
                m.school,
                m.singer,
                COUNT(*) as interaction_count,
                CAST(SUM(CASE 
                    WHEN mc.sentiment IN ('积极', '非常积极') THEN 2
                    WHEN mc.sentiment = '中性' THEN 1
                    ELSE 0
                END) AS DECIMAL(10,2)) as sentiment_score,
                CAST(AVG(mc.like_count) AS DECIMAL(10,2)) as avg_likes
            FROM music_comments mc
            JOIN music m ON mc.music_id = m.id
            WHERE mc.user_id = %s
            GROUP BY m.languages, m.school, m.singer
            HAVING interaction_count >= 2
        """
        cursor.execute(preference_sql, (user_id,))
        preferences = cursor.fetchall()
        
        if preferences:
            # 2. 构建用户偏好权重
            language_weights = {}
            school_weights = {}
            singer_weights = {}
            
            for lang, school, singer, count, sentiment, likes in preferences:
                # 将 Decimal 转换为 float
                count = float(count)
                sentiment = float(sentiment) if sentiment else 0.0
                likes = float(likes) if likes else 0.0
                
                if lang:
                    weight = (count * 0.4 + sentiment * 0.4 + likes * 0.2) / (count * 2)
                    language_weights[lang] = min(weight, 1.0)
                if school:
                    weight = (count * 0.4 + sentiment * 0.4 + likes * 0.2) / (count * 2)
                    school_weights[school] = min(weight, 1.0)
                if singer:
                    weight = (count * 0.4 + sentiment * 0.4 + likes * 0.2) / (count * 2)
                    singer_weights[singer] = min(weight, 1.0)
            
            # 如果没有任何权重，使用默认推荐逻辑
            if not (language_weights or school_weights or singer_weights):
                preferences = []
            else:
                # 构建权重计算表达式
                weight_expressions = []
                params = []
                
                if language_weights:
                    placeholders = ','.join(['%s'] * len(language_weights))
                    weight_expr = f"(CASE WHEN m.languages IN ({placeholders}) THEN {' + '.join([str(float(w * 15)) for w in language_weights.values()])} ELSE 0 END)"
                    weight_expressions.append(weight_expr)
                    params.extend(language_weights.keys())
                
                if school_weights:
                    placeholders = ','.join(['%s'] * len(school_weights))
                    weight_expr = f"(CASE WHEN m.school IN ({placeholders}) THEN {' + '.join([str(float(w * 15)) for w in school_weights.values()])} ELSE 0 END)"
                    weight_expressions.append(weight_expr)
                    params.extend(school_weights.keys())
                
                if singer_weights:
                    placeholders = ','.join(['%s'] * len(singer_weights))
                    weight_expr = f"(CASE WHEN m.singer IN ({placeholders}) THEN {' + '.join([str(float(w * 15)) for w in singer_weights.values()])} ELSE 0 END)"
                    weight_expressions.append(weight_expr)
                    params.extend(singer_weights.keys())
                
                # 构建完整的权重计算表达式
                preference_weights = ' + '.join(weight_expressions)
                
                # 构建基础分数计算SQL
                base_score_sql = f"""
                    {preference_weights} +
                    (CAST(m.play_count AS DECIMAL(10,2)) / NULLIF((SELECT CAST(MAX(play_count) AS DECIMAL(10,2)) FROM music), 0) * 10) +
                    (CAST(m.like_count AS DECIMAL(10,2)) / NULLIF((SELECT CAST(MAX(like_count) AS DECIMAL(10,2)) FROM music), 0) * 15) +
                    (CAST(m.download_count AS DECIMAL(10,2)) / NULLIF((SELECT CAST(MAX(download_count) AS DECIMAL(10,2)) FROM music), 0) * 8) +
                    (CAST(m.comment_count AS DECIMAL(10,2)) / NULLIF((SELECT CAST(MAX(comment_count) AS DECIMAL(10,2)) FROM music), 0) * 12)
                """
                
                # 3. 获取用户未听过的音乐
                recommendation_sql = """
                    SELECT 
                        m.id, m.music_name, m.singer, m.languages, m.school,
                        m.record_company, m.release_date, m.music_url, m.image_url,
                        m.play_count, m.like_count, m.download_count, m.comment_count,
                        CAST(
                            (
                                {}
                            ) * 0.75 AS DECIMAL(10,2)
                        ) as base_score,
                        DATEDIFF(CURRENT_DATE, m.release_date) as days_since_release
                    FROM music m
                    WHERE m.id NOT IN (
                        SELECT music_id 
                        FROM music_comments 
                        WHERE user_id = %s
                    )
                    HAVING base_score > 0
                    ORDER BY base_score DESC
                    LIMIT 20
                """.format(base_score_sql)
                
                # 添加用户ID到参数列表
                params.append(user_id)
                
                cursor.execute(recommendation_sql, params)
        
        if not preferences:
            # 4. 如果没有用户偏好，返回热门和新音乐的混合推荐
            cursor.execute("""
                SELECT * FROM (
                    SELECT 
                        id, music_name, singer, languages, school,
                        record_company, release_date, music_url, image_url,
                        play_count, like_count, download_count, comment_count,
                        CAST(
                            CASE 
                                WHEN DATEDIFF(CURRENT_DATE, release_date) <= 30 THEN
                                    30 + CAST((play_count + like_count * 2 + download_count + comment_count * 1.5) AS DECIMAL(10,2)) / 
                                    CAST((SELECT MAX(play_count + like_count * 2 + download_count + comment_count * 1.5) FROM music) AS DECIMAL(10,2)) * 35
                                ELSE
                                    CAST((play_count + like_count * 2 + download_count + comment_count * 1.5) AS DECIMAL(10,2)) / 
                                    CAST((SELECT MAX(play_count + like_count * 2 + download_count + comment_count * 1.5) FROM music) AS DECIMAL(10,2)) * 65
                            END
                        AS DECIMAL(10,2)) * 0.75 as base_score,
                        DATEDIFF(CURRENT_DATE, release_date) as days_since_release,
                        @rank := IF(
                            DATEDIFF(CURRENT_DATE, release_date) <= 30,
                            @new_rank := @new_rank + 1,
                            @pop_rank := @pop_rank + 1
                        ) as rank,
                        IF(DATEDIFF(CURRENT_DATE, release_date) <= 30, 'new', 'popular') as type
                    FROM music,
                        (SELECT @new_rank := 0, @pop_rank := 0) r
                    ORDER BY 
                        type,
                        CASE 
                            WHEN DATEDIFF(CURRENT_DATE, release_date) <= 30 THEN
                                play_count + like_count * 2 + download_count + comment_count * 1.5
                            ELSE
                                play_count + like_count * 2 + download_count + comment_count * 1.5
                        END DESC
                ) ranked
                WHERE rank <= 10
                ORDER BY base_score DESC
                LIMIT 20
            """)
        
        results = cursor.fetchall()
        
        # 格式化推荐结果
        recommendations = []
        for row in results:
            # 处理日期字段
            release_date = None
            if row[6]:
                try:
                    if hasattr(row[6], 'strftime'):
                        release_date = row[6].strftime('%Y-%m-%d')
                    elif isinstance(row[6], str):
                        release_date = row[6]
                except Exception as e:
                    print(f"日期格式化错误: {str(e)}")
                    release_date = str(row[6])
            
            # 计算最终推荐分数
            base_score = float(row[13]) if row[13] else 0
            days_since_release = float(row[14]) if row[14] else 0
            
            # 根据发布时间调整分数
            time_factor = 1.0
            if days_since_release <= 30:  # 新歌加分
                time_factor = 1.1
            elif days_since_release > 365:  # 老歌稍微降分
                time_factor = 0.95
            
            # 计算最终分数（不使用随机因素）
            final_score = base_score * time_factor
            
            # 将分数映射到不同区间
            if final_score >= 80:
                final_score = 90  # 极力推荐
            elif final_score >= 70:
                final_score = 80  # 强烈推荐
            elif final_score >= 60:
                final_score = 70  # 推荐
            elif final_score >= 50:
                final_score = 60  # 建议尝试
            elif final_score >= 40:
                final_score = 50  # 可以考虑
            else:
                final_score = 40  # 供参考
            
            recommendations.append({
                'id': row[0],
                'music_name': row[1],
                'singer': row[2],
                'languages': row[3],
                'school': row[4],
                'record_company': row[5],
                'release_date': release_date,
                'music_url': row[7],
                'image_url': row[8],
                'play_count': int(row[9]) if row[9] else 0,
                'like_count': int(row[10]) if row[10] else 0,
                'download_count': int(row[11]) if row[11] else 0,
                'comment_count': int(row[12]) if row[12] else 0,
                'score': final_score
            })
        
        # 按分数降序排序
        recommendations.sort(key=lambda x: (x['score'], x['play_count']), reverse=True)
        return recommendations
    except Exception as e:
        print(f"获取推荐失败: {str(e)}")
        return []
    finally:
        cursor.close()
        conn.close()

# 删除评论
def delete_comment(comment_id, music_id):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # 开始事务
        conn.begin()
        
        # 删除评论
        sql1 = "DELETE FROM music_comments WHERE id = %s"
        cursor.execute(sql1, (comment_id,))
        
        # 更新音乐评论数
        sql2 = "UPDATE music SET comment_count = comment_count - 1 WHERE id = %s"
        cursor.execute(sql2, (music_id,))
        
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"删除评论失败: {str(e)}")
        return False
    finally:
        cursor.close()
        conn.close()

class MusicDao:
    @staticmethod
    def add_play_history(user_id, music_id):
        conn = get_connection()
        cursor = conn.cursor()
        try:
            # 开始事务
            conn.begin()
            
            # 添加播放历史记录
            sql1 = "INSERT INTO play_history (user_id, music_id) VALUES (%s, %s)"
            cursor.execute(sql1, (user_id, music_id))
            
            # 更新音乐播放次数
            sql2 = "UPDATE music SET play_count = play_count + 1 WHERE id = %s"
            cursor.execute(sql2, (music_id,))
            
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            print(f"添加播放历史失败: {str(e)}")
            return False
        finally:
            cursor.close()
            conn.close()

    @staticmethod
    def get_play_history(user_id, page=1, limit=20):
        conn = get_connection()
        cursor = conn.cursor()
        try:
            # 计算偏移量
            offset = (page - 1) * limit
            
            # 查询总数
            count_sql = "SELECT COUNT(*) FROM play_history WHERE user_id = %s"
            cursor.execute(count_sql, (user_id,))
            total = cursor.fetchone()[0]
            
            # 查询播放历史数据
            sql = """
                SELECT 
                    m.id, m.music_name, m.singer, m.languages, m.school,
                    m.record_company, m.release_date, m.music_url, m.image_url,
                    m.play_count, m.like_count, m.download_count, m.comment_count,
                    ph.play_time
                FROM play_history ph
                JOIN music m ON ph.music_id = m.id
                WHERE ph.user_id = %s
                ORDER BY ph.play_time DESC
                LIMIT %s, %s
            """
            cursor.execute(sql, (user_id, offset, limit))
            data = cursor.fetchall()
            
            # 格式化返回数据
            history_list = []
            for row in data:
                release_date = None
                if row[6]:
                    try:
                        if hasattr(row[6], 'strftime'):
                            release_date = row[6].strftime('%Y-%m-%d')
                        elif isinstance(row[6], str):
                            release_date = row[6]
                    except Exception as e:
                        print(f"日期格式化错误: {str(e)}")
                        release_date = str(row[6])
                
                play_time = None
                if row[13]:
                    try:
                        if hasattr(row[13], 'strftime'):
                            play_time = row[13].strftime('%Y-%m-%d %H:%M:%S')
                        elif isinstance(row[13], str):
                            play_time = row[13]
                    except Exception as e:
                        print(f"日期格式化错误: {str(e)}")
                        play_time = str(row[13])
                
                history_list.append({
                    'id': row[0],
                    'music_name': row[1],
                    'singer': row[2],
                    'languages': row[3],
                    'school': row[4],
                    'record_company': row[5],
                    'release_date': release_date,
                    'music_url': row[7],
                    'image_url': row[8],
                    'play_count': int(row[9]) if row[9] else 0,
                    'like_count': int(row[10]) if row[10] else 0,
                    'download_count': int(row[11]) if row[11] else 0,
                    'comment_count': int(row[12]) if row[12] else 0,
                    'play_time': play_time
                })
            
            return {
                'total': total,
                'data': history_list
            }
        except Exception as e:
            print(f"获取播放历史失败: {str(e)}")
            return {
                'total': 0,
                'data': []
            }
        finally:
            cursor.close()
            conn.close()