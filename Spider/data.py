import random
from datetime import datetime, timedelta
from Dao.MusicDao import get_connection
import pandas as pd

def generate_play_history(num_records=1000):
    """
    生成测试用的播放历史数据
    
    参数:
        num_records: 要生成的记录数量
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # 获取所有用户ID
        cursor.execute("SELECT id FROM py_user")
        user_ids = [row[0] for row in cursor.fetchall()]
        
        # 获取所有音乐ID及其热度信息
        cursor.execute("""
            SELECT 
                id, 
                play_count,
                like_count,
                download_count,
                comment_count
            FROM music
        """)
        music_data = cursor.fetchall()
        
        if not user_ids or not music_data:
            print("没有找到用户或音乐数据")
            return
        
        # 计算音乐热度分数
        music_scores = {}
        for music in music_data:
            music_id = music[0]
            # 计算热度分数：播放次数*0.4 + 点赞数*0.3 + 下载数*0.2 + 评论数*0.1
            score = (
                (music[1] or 0) * 0.4 +  # play_count
                (music[2] or 0) * 0.3 +  # like_count
                (music[3] or 0) * 0.2 +  # download_count
                (music[4] or 0) * 0.1    # comment_count
            )
            music_scores[music_id] = score
        
        # 生成播放记录
        play_records = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # 生成最近30天的数据
        
        for _ in range(num_records):
            user_id = random.choice(user_ids)
            # 根据热度分数选择音乐
            music_id = random.choices(
                list(music_scores.keys()),
                weights=list(music_scores.values()),
                k=1
            )[0]
            
            # 生成随机播放时间
            random_days = random.randint(0, 30)
            random_hours = random.randint(0, 23)
            random_minutes = random.randint(0, 59)
            play_time = start_date + timedelta(
                days=random_days,
                hours=random_hours,
                minutes=random_minutes
            )
            
            play_records.append({
                'user_id': user_id,
                'music_id': music_id,
                'play_time': play_time
            })
        
        # 将数据转换为DataFrame
        df = pd.DataFrame(play_records)
        
        # 按用户ID和音乐ID分组，计算每个用户对每首歌的播放次数
        grouped = df.groupby(['user_id', 'music_id']).size().reset_index(name='play_count')
        
        # 为每个用户-音乐对生成多条播放记录
        final_records = []
        for _, row in grouped.iterrows():
            user_id = row['user_id']
            music_id = row['music_id']
            play_count = row['play_count']
            
            # 为每次播放生成一条记录
            for i in range(play_count):
                # 生成随机播放时间
                random_days = random.randint(0, 30)
                random_hours = random.randint(0, 23)
                random_minutes = random.randint(0, 59)
                play_time = start_date + timedelta(
                    days=random_days,
                    hours=random_hours,
                    minutes=random_minutes
                )
                
                final_records.append((
                    user_id,
                    music_id,
                    play_time
                ))
        
        # 插入数据到数据库
        insert_query = """
            INSERT INTO play_history (user_id, music_id, play_time)
            VALUES (%s, %s, %s)
        """
        cursor.executemany(insert_query, final_records)
        conn.commit()
        
        print(f"成功生成 {len(final_records)} 条播放记录")
        
    except Exception as e:
        print(f"生成播放记录失败: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def generate_user_play_patterns():
    """
    生成更真实的用户播放模式
    - 某些用户会重复播放特定类型的音乐
    - 某些用户会频繁播放特定歌手的作品
    - 某些用户会按照时间段有规律的播放习惯
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # 获取所有用户ID
        cursor.execute("SELECT id FROM py_user")
        user_ids = [row[0] for row in cursor.fetchall()]
        
        # 获取音乐数据（包含语种、流派、歌手信息）
        cursor.execute("""
            SELECT 
                id, 
                languages, 
                school, 
                singer,
                play_count,
                like_count,
                download_count,
                comment_count
            FROM music 
            WHERE languages IS NOT NULL 
            AND school IS NOT NULL 
            AND singer IS NOT NULL
        """)
        music_data = cursor.fetchall()
        
        if not user_ids or not music_data:
            print("没有找到用户或音乐数据")
            return
        
        # 为每个用户生成播放记录
        for user_id in user_ids:
            # 随机选择用户的音乐偏好
            preferred_languages = random.sample(
                list(set(m[1] for m in music_data)), 
                random.randint(1, 3)
            )
            preferred_schools = random.sample(
                list(set(m[2] for m in music_data)), 
                random.randint(1, 3)
            )
            preferred_singers = random.sample(
                list(set(m[3] for m in music_data)), 
                random.randint(1, 5)
            )
            
            # 生成该用户的播放记录
            user_records = []
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            # 生成100-200条播放记录
            num_records = random.randint(100, 200)
            
            for _ in range(num_records):
                # 70%的概率选择偏好音乐，30%的概率随机选择
                if random.random() < 0.7:
                    # 从偏好中选择
                    preferred_music = [
                        m for m in music_data 
                        if m[1] in preferred_languages 
                        or m[2] in preferred_schools 
                        or m[3] in preferred_singers
                    ]
                    if preferred_music:
                        # 根据热度选择音乐
                        music_scores = {
                            m[0]: (
                                (m[4] or 0) * 0.4 +  # play_count
                                (m[5] or 0) * 0.3 +  # like_count
                                (m[6] or 0) * 0.2 +  # download_count
                                (m[7] or 0) * 0.1    # comment_count
                            )
                            for m in preferred_music
                        }
                        music_id = random.choices(
                            list(music_scores.keys()),
                            weights=list(music_scores.values()),
                            k=1
                        )[0]
                    else:
                        music_id = random.choice(music_data)[0]
                else:
                    # 随机选择
                    music_id = random.choice(music_data)[0]
                
                # 生成播放时间（模拟用户通常在特定时间段听音乐）
                hour = random.randint(8, 23)  # 8:00-23:00
                minute = random.randint(0, 59)
                random_days = random.randint(0, 30)
                
                play_time = start_date + timedelta(
                    days=random_days,
                    hours=hour,
                    minutes=minute
                )
                
                user_records.append((
                    user_id,
                    music_id,
                    play_time
                ))
            
            # 插入该用户的播放记录
            insert_query = """
                INSERT INTO play_history (user_id, music_id, play_time)
                VALUES (%s, %s, %s)
            """
            cursor.executemany(insert_query, user_records)
            conn.commit()
            
            print(f"为用户 {user_id} 生成 {len(user_records)} 条播放记录")
        
    except Exception as e:
        print(f"生成用户播放模式失败: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def verify_play_history():
    """
    验证生成的播放历史数据
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # 验证记录总数
        cursor.execute("SELECT COUNT(*) FROM play_history")
        total_records = cursor.fetchone()[0]
        print(f"总播放记录数: {total_records}")
        
        # 验证用户分布
        cursor.execute("""
            SELECT user_id, COUNT(*) as play_count 
            FROM play_history 
            GROUP BY user_id 
            ORDER BY play_count DESC 
            LIMIT 5
        """)
        print("\n播放次数最多的前5个用户:")
        for user_id, count in cursor.fetchall():
            print(f"用户ID: {user_id}, 播放次数: {count}")
        
        # 验证音乐分布
        cursor.execute("""
            SELECT 
                m.id,
                m.music_name,
                m.singer,
                COUNT(ph.id) as play_count
            FROM play_history ph
            JOIN music m ON ph.music_id = m.id
            GROUP BY m.id, m.music_name, m.singer
            ORDER BY play_count DESC 
            LIMIT 5
        """)
        print("\n播放次数最多的前5首音乐:")
        for music_id, music_name, singer, count in cursor.fetchall():
            print(f"音乐ID: {music_id}, 歌名: {music_name}, 歌手: {singer}, 播放次数: {count}")
        
        # 验证时间分布
        cursor.execute("""
            SELECT 
                HOUR(play_time) as hour,
                COUNT(*) as play_count
            FROM play_history
            GROUP BY HOUR(play_time)
            ORDER BY hour
        """)
        print("\n各时段的播放分布:")
        for hour, count in cursor.fetchall():
            print(f"{hour:02d}:00 - {hour:02d}:59: {count}次播放")
        
    except Exception as e:
        print(f"验证数据失败: {str(e)}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    # 生成基础播放记录
    generate_play_history(1000)
    
    # 生成更真实的用户播放模式
    generate_user_play_patterns()
    
    # 验证生成的数据
    verify_play_history()
