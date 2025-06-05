import random
from datetime import datetime, timedelta
import pymysql  # 替换为 PyMySQL
import time

# 数据库配置
db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '123456',
    'database': 'py_music',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

# 评论模板
positive_comments = [
    # 情感类评论
    "这首歌真的很棒！",
    "循环播放中，百听不厌",
    "歌词写得太走心了",
    "这个旋律太抓耳了",
    "声音真的很有特色",
    "超级喜欢这首歌",
    "每次听都有不同的感触",
    "单曲循环中...",
    "听哭了",
    "这歌词说到我心坎里了",
    
    # 音乐专业性评论
    "编曲非常用心",
    "这个编曲太神了",
    "鼓点很带感",
    "前奏就抓住我了",
    "这个转音绝了",
    "高音真的震撼",
    "这个和声编排很棒",
    "音色很温暖",
    "这个混音很专业",
    
    # 场景联想类
    "适合夜晚一个人听",
    "开车必备歌曲",
    "下雨天循环播放",
    "失恋的时候就要听这个",
    "通勤路上单曲循环",
    "熬夜必备良曲",
    "春天就要听这样的歌",
    "适合一个人发呆的时候听",
    
    # 情绪类
    "听得我热泪盈眶",
    "太治愈了",
    "心都要化了",
    "听得我起鸡皮疙瘩",
    "浑身都是故事感",
    "听得我想谈恋爱了",
    "心都被暖化了"
]

detailed_comments = [
    # 音乐专业点评
    "这首歌的编曲真的很精致，每一个音符都恰到好处，尤其是副歌部分的弦乐编排",
    "歌手的情感表达太到位了，听得我起鸡皮疙瘩，这种共鸣感太强了",
    "前奏一响就被吸引住了，整首歌的制作水平都很高，混音师功力很深",
    "这个编曲层次分明，主歌铺垫，副歌爆发，结尾余韵悠长，结构很完整",
    
    # 情感共鸣
    "这首歌让我想起了很多往事，感谢音乐的治愈，有些故事只有音乐才能诉说",
    "副歌部分太惊艳了，一下就抓住了听众的心，听得我泪目了",
    "歌词写得特别走心，像是在讲述我的故事，每个字都戳中内心",
    "这个歌手的声线太有特色了，情感表达力很强，听得我都要哭了",
    
    # 场景描述
    "深夜一个人听这首歌特别有感觉，仿佛整个世界都安静下来了",
    "下雨天听这首歌最配了，坐在窗边，一杯咖啡，完美的享受",
    "通勤路上单曲循环，感觉整个世界都变得美好起来",
    "失恋的时候循环播放，眼泪不知不觉就流下来了",
    
    # 音乐元素点评
    "这首歌的旋律线条优美，和声编排很讲究，尤其是副歌部分的层次感",
    "编曲中若隐若现的钢琴声很有意境，为整首歌增添了不少感染力",
    "前奏的弦乐编排很有电影感，一开始就把人带入意境了",
    "这个制作很讲究，鼓点的力度，贝斯的走向，都恰到好处"
]

def generate_comment():
    """生成一条随机评论"""
    if random.random() < 0.7:  # 70%概率生成简单评论
        comment = random.choice(positive_comments)
        if random.random() < 0.4:  # 40%概率添加额外内容
            comment += "，" + random.choice([
                "好听！", 
                "推荐！", 
                "收藏了！", 
                "循环播放中~",
                "太赞了！",
                "忍不住分享！",
                "必须打call！",
                "爱了爱了！",
                "这才是神曲啊！",
                "这个必须顶！"
            ])
            
        # 20%概率添加emoji
        if random.random() < 0.2:
            comment += random.choice([
                " ❤️", " 👍", " 🎵", " ✨", " 💗",
                " 🌟", " 💖", " 🎶", " 🥰", " 💝"
            ])
    else:  # 30%概率生成详细评论
        comment = random.choice(detailed_comments)
        
        # 15%概率添加个性化时间标记
        if random.random() < 0.15:
            comment += random.choice([
                "\n深夜听更有感觉",
                "\n单曲循环第999遍",
                "\n留个评论记录一下此刻的心情",
                "\n希望你们也能喜欢这首歌",
                "\n分享给同样喜欢音乐的你们"
            ])
    
    return comment

def generate_timestamp():
    """生成随机时间戳（最近3个月内）"""
    now = datetime.now()
    days_ago = random.randint(0, 90)
    hours = random.randint(0, 23)      # 随机小时
    minutes = random.randint(0, 59)    # 随机分钟
    seconds = random.randint(0, 59)    # 随机秒
    
    random_date = now - timedelta(days=days_ago)
    # 替换时间部分
    random_date = random_date.replace(
        hour=hours,
        minute=minutes,
        second=seconds
    )
    
    return random_date.strftime('%Y-%m-%d %H:%M:%S')

def generate_likes(comment_length, is_detailed=False):
    """
    生成评论的点赞数
    参数:
        comment_length: 评论的长度
        is_detailed: 是否是详细评论
    """
    # 基础点赞范围
    if is_detailed:
        # 详细评论倾向于获得更多点赞
        base_likes = random.randint(5, 50)
    else:
        base_likes = random.randint(0, 20)
    
    # 根据评论长度增加点赞概率
    length_bonus = min(comment_length // 10, 30)
    
    # 10%概率成为热门评论（获得更多点赞）
    if random.random() < 0.1:
        hot_bonus = random.randint(50, 200)
    else:
        hot_bonus = 0
    
    return base_likes + length_bonus + hot_bonus

def test_connection():
    """测试数据库连接"""
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            conn = pymysql.connect(**db_config)
            print("数据库连接成功！")
            cursor = conn.cursor()
            
            # 检查数据库字符集
            cursor.execute("SHOW VARIABLES LIKE 'character_set_database'")
            print("数据库字符集:", cursor.fetchone())
            
            # 创建表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS music_comments (
                    id int(11) NOT NULL AUTO_INCREMENT,
                    music_id int(11) NOT NULL,
                    user_id int(11) NOT NULL,
                    comment_content text,
                    comment_time datetime DEFAULT NULL,
                    like_count int(11) DEFAULT '0',
                    PRIMARY KEY (id),
                    KEY idx_music_id (music_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            conn.commit()
            print("数据表检查/创建成功！")
            
            cursor.close()
            conn.close()
            return True
            
        except pymysql.Error as err:
            print(f"尝试 {retry_count + 1}/{max_retries} 失败: {err}")
            retry_count += 1
            time.sleep(1)
            
        except Exception as e:
            print(f"发生未知错误: {e}")
            return False
            
    print("达到最大重试次数，连接失败")
    return False

def insert_comments():
    """为每首歌插入评论"""
    if not test_connection():
        print("数据库连接测试失败，请检查配置！")
        return
        
    conn = None
    cursor = None
    try:
        conn = pymysql.connect(**db_config)
        cursor = conn.cursor()
        
        # 获取所有音乐ID
        cursor.execute("SELECT id, music_name FROM music")
        songs = cursor.fetchall()
        
        if not songs:
            print("没有找到任何音乐记录！")
            return
            
        # 为每首歌生成评论
        for song in songs:
            try:
                song_id = song['id']
                song_name = song['music_name']
                
                # 随机生成50-120条评论
                num_comments = random.randint(50, 120)
                
                # 准备评论数据
                comments_data = []
                for _ in range(num_comments):
                    comment = generate_comment()
                    timestamp = generate_timestamp()
                    user_id = random.randint(1, 100)
                    
                    is_detailed = len(comment) > 50
                    likes = generate_likes(len(comment), is_detailed)
                    
                    comments_data.append((song_id, user_id, comment, timestamp, likes))
                
                # 批量插入评论
                insert_query = """
                INSERT INTO music_comments 
                (music_id, user_id, comment_content, comment_time, like_count,sentiment) 
                VALUES (%s, %s, %s, %s, %s, NULL)
                """
                cursor.executemany(insert_query, comments_data)
                conn.commit()
                
                print(f"已为歌曲 '{song_name}' (ID: {song_id}) 生成 {num_comments} 条评论")
                
            except pymysql.Error as err:
                print(f"处理歌曲 '{song_name}' 时出错: {err}")
                if conn:
                    conn.rollback()
                continue
        
    except pymysql.Error as err:
        print(f"数据库错误: {err}")
    finally:
        try:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
                print("数据库连接已关闭")
        except pymysql.Error as err:
            print(f"关闭连接时出错: {err}")

if __name__ == "__main__":
    insert_comments()
