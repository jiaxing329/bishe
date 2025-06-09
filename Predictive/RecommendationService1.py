import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
import pandas as pd
from Dao.MusicDao import get_connection
import os
from collections import defaultdict
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col, expr
from hdfs import InsecureClient
from mrjob.job import MRJob
from mrjob.step import MRStep
import json
from typing import List, Dict, Any


class RecommendationService:
    def __init__(self, use_spark=False):
        """初始化推荐服务"""
        self.use_spark = use_spark
        if use_spark:
            try:
                self._init_spark()
            except Exception as e:
                print(f"Spark初始化失败: {str(e)}")
                self.use_spark = False
        self.model = None
        self.tokenizer = None
        self.max_sequence_length = 100
        self.embedding_dim = 100
        self.vocab_size = 10000

        # 加载或创建模型
        self._load_or_create_model()

    def _init_spark(self):
        """初始化Spark"""
        try:
            # 获取当前文件所在目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            mysql_jar = os.path.join(current_dir, "..", "lib", "mysql-connector-java-5.1.49.jar")

            # 初始化Spark会话
            self.spark = SparkSession.builder \
                .appName("MusicRecommendation") \
                .config("spark.jars", mysql_jar) \
                .config("spark.executor.memory", "2g") \
                .config("spark.driver.memory", "2g") \
                .getOrCreate()

            print("Spark初始化成功")

        except Exception as e:
            print(f"Spark初始化失败: {str(e)}")
            raise

    def _load_or_create_model(self):
        """加载或创建LSTM模型"""
        model_path = 'models/lstm_recommendation.h5'
        tokenizer_path = 'models/tokenizer.pkl'

        if os.path.exists(model_path) and os.path.exists(tokenizer_path):
            try:
                self.model = load_model(model_path)
                with open(tokenizer_path, 'rb') as f:
                    self.tokenizer = pickle.load(f)
                print("成功加载现有模型")
                return
            except Exception as e:
                print(f"加载模型失败: {str(e)}")

        # 创建新模型
        self._create_and_train_model()

    def _create_and_train_model(self):
        """创建并训练LSTM模型"""
        try:
            # 获取训练数据
            conn = get_connection()
            cursor = conn.cursor()

            # 获取用户评论数据
            cursor.execute("""
                SELECT 
                    mc.user_id,
                    mc.music_id,
                    mc.comment_content,
                    mc.sentiment,
                    m.languages,
                    m.school,
                    m.singer
                FROM music_comments mc
                JOIN music m ON mc.music_id = m.id
                WHERE mc.comment_content IS NOT NULL
            """)

            comments_data = cursor.fetchall()

            if not comments_data:
                raise Exception("没有找到训练数据")

            # 准备文本数据
            texts = [row[2] for row in comments_data]

            # 创建并训练tokenizer
            self.tokenizer = Tokenizer(num_words=self.vocab_size)
            self.tokenizer.fit_on_texts(texts)

            # 转换文本为序列
            sequences = self.tokenizer.texts_to_sequences(texts)
            X = pad_sequences(sequences, maxlen=self.max_sequence_length)

            # 准备标签（使用情感作为标签）
            y = np.array([1 if row[3] == '积极' else 0 for row in comments_data])

            # 创建LSTM模型
            self.model = Sequential([
                Embedding(self.vocab_size, self.embedding_dim,
                          input_length=self.max_sequence_length),
                LSTM(128, return_sequences=True),
                Dropout(0.3),
                LSTM(64),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])

            # 编译模型
            self.model.compile(optimizer='adam',
                               loss='binary_crossentropy',
                               metrics=['accuracy'])

            # 训练模型
            self.model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2)

            # 保存模型和tokenizer
            self.model.save('models/lstm_recommendation.h5')
            with open('models/tokenizer.pkl', 'wb') as f:
                pickle.dump(self.tokenizer, f)

            print("模型训练完成并保存")

        except Exception as e:
            print(f"创建模型失败: {str(e)}")
            raise
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()

    def get_music_features(self, music_id):
        """获取音乐特征"""
        conn = get_connection()
        try:
            query = """
                SELECT languages, school, play_count, like_count, comment_count
                FROM music 
                WHERE id = %s
            """
            df = pd.read_sql(query, conn, params=[music_id])

            # 添加统计特征
            features = pd.get_dummies(df[['languages', 'school']])

            # 归一化统计数据
            for col in ['play_count', 'like_count', 'comment_count']:
                max_val = df[col].max()
                features[col] = df[col] / max_val if max_val > 0 else 0

            # 确保所有特征列都存在
            for col in self.feature_columns:
                if col not in features.columns:
                    features[col] = 0
            return features[self.feature_columns].values[0]
        finally:
            conn.close()

    def get_user_preferences(self, user_id):
        """获取用户偏好"""
        conn = get_connection()
        try:
            # 获取用户评论过的音乐及其评论内容
            query = """
                SELECT 
                    m.id,
                    m.languages,
                    m.school,
                    m.play_count,
                    m.like_count,
                    m.comment_count,
                    COUNT(mc.id) as user_comment_count,
                    GROUP_CONCAT(mc.comment_content) as all_comments
                FROM music m
                INNER JOIN music_comments mc ON m.id = mc.music_id
                WHERE mc.user_id = %s
                GROUP BY m.id, m.languages, m.school, m.play_count, m.like_count, m.comment_count
            """
            df = pd.read_sql(query, conn, params=[user_id])

            if df.empty:
                return None

            # 计算用户对每种类型音乐的偏好权重
            preferences = {}

            # 语种偏好
            language_weights = df.groupby('languages').agg({
                'user_comment_count': 'sum',
                'play_count': 'mean',
                'like_count': 'mean'
            })
            language_weights['total_weight'] = (
                    0.5 * language_weights['user_comment_count'] / language_weights['user_comment_count'].max() +
                    0.3 * language_weights['play_count'] / language_weights['play_count'].max() +
                    0.2 * language_weights['like_count'] / language_weights['like_count'].max()
            )
            preferences['languages'] = language_weights['total_weight'].to_dict()

            # 流派偏好
            school_weights = df.groupby('school').agg({
                'user_comment_count': 'sum',
                'play_count': 'mean',
                'like_count': 'mean'
            })
            school_weights['total_weight'] = (
                    0.5 * school_weights['user_comment_count'] / school_weights['user_comment_count'].max() +
                    0.3 * school_weights['play_count'] / school_weights['play_count'].max() +
                    0.2 * school_weights['like_count'] / school_weights['like_count'].max()
            )
            preferences['school'] = school_weights['total_weight'].to_dict()

            return preferences
        finally:
            conn.close()

    def calculate_similarity(self, user_preferences, music):
        """计算用户偏好与音乐的相似度"""
        score = 0

        # 语种匹配度
        if music['languages'] in user_preferences['languages']:
            score += user_preferences['languages'][music['languages']] * 0.6

        # 流派匹配度
        if music['school'] in user_preferences['school']:
            score += user_preferences['school'][music['school']] * 0.4

        return score

    def get_popular_music(self, limit=10):
        """获取热门音乐"""
        conn = get_connection()
        try:
            query = """
                SELECT id, music_name, singer, languages, school, 
                       image_url, play_count, like_count, comment_count
                FROM music
                ORDER BY (play_count + like_count * 2 + comment_count * 3) DESC
                LIMIT %s
            """
            df = pd.read_sql(query, conn, params=[limit])

            # 计算综合得分
            df['score'] = (df['play_count'] + df['like_count'] * 2 + df['comment_count'] * 3)
            max_score = df['score'].max()
            df['score'] = df['score'] / max_score if max_score > 0 else 0

            return df.to_dict('records')
        finally:
            conn.close()

    def get_spark_recommendations(self, user_id, limit=12):
        """使用Spark ALS算法获取推荐"""
        try:
            # 从MySQL读取数据
            ratings_df = self.spark.read \
                .format("jdbc") \
                .option("url", "jdbc:mysql://localhost:3306/py_music") \
                .option("driver", "com.mysql.jdbc.Driver") \
                .option("dbtable", """
                    (SELECT 
                        user_id,
                        music_id,
                        CASE 
                            WHEN sentiment = '积极' THEN 5
                            WHEN sentiment = '消极' THEN 1
                            ELSE 3
                        END as rating
                    FROM music_comments) as ratings
                """) \
                .option("user", "root") \
                .option("password", "123456") \
                .load()

            # 创建ALS模型
            als = ALS(maxIter=5,
                      regParam=0.01,
                      userCol="user_id",
                      itemCol="music_id",
                      ratingCol="rating",
                      coldStartStrategy="drop",
                      nonnegative=True)

            # 训练模型
            model = als.fit(ratings_df)

            # 获取用户未评论过的音乐
            user_unrated_df = self.spark.read \
                .format("jdbc") \
                .option("url", "jdbc:mysql://localhost:3306/py_music") \
                .option("driver", "com.mysql.jdbc.Driver") \
                .option("dbtable", f"""
                    (SELECT DISTINCT id as music_id
                     FROM music 
                     WHERE id NOT IN (
                         SELECT music_id 
                         FROM music_comments 
                         WHERE user_id = {user_id}
                     )) as unrated
                """) \
                .option("user", "root") \
                .option("password", "123456") \
                .load()

            # 生成推荐
            user_recs = model.transform(
                user_unrated_df.withColumn("user_id", expr(f"int({user_id})"))
            )

            # 获取推荐结果
            recommendations = user_recs.orderBy("prediction", ascending=False) \
                .limit(limit) \
                .join(
                self.spark.read \
                    .format("jdbc") \
                    .option("url", "jdbc:mysql://localhost:3306/py_music") \
                    .option("driver", "com.mysql.jdbc.Driver") \
                    .option("dbtable", "music") \
                    .option("user", "root") \
                    .option("password", "123456") \
                    .load(),
                    col("music_id") == col("id")
            ) \
                .select(
                "id", "music_name", "singer", "languages",
                "school", "image_url",
                (col("prediction") * 20).alias("score")
            ) \
                .collect()

            # 格式化结果
            return [{
                'id': row.id,
                'music_name': row.music_name,
                'singer': row.singer,
                'languages': row.languages or '其他',
                'school': row.school or '其他',
                'image_url': row.image_url,
                'score': round(float(row.score), 1)
            } for row in recommendations]

        except Exception as e:
            print(f"Spark推荐生成错误: {str(e)}")
            return []

    def get_recommendations(self, user_id, music_id=None, limit=12):
        """获取综合推荐"""
        if music_id:
            # 如果指定了音乐ID，优先返回关联推荐
            associated_recs = self.get_associated_recommendations(music_id, limit)
            if associated_recs:
                return associated_recs

        # 否则返回基于用户的推荐
        if self.use_spark:
            return self.get_spark_recommendations(user_id, limit)
        else:
            try:
                conn = get_connection()
                cursor = conn.cursor()

                # 1. 获取用户的播放历史
                cursor.execute("""
                    SELECT 
                        m.id,
                        m.music_name,
                        m.singer,
                        m.languages,
                        m.school,
                        ph.play_time
                    FROM play_history ph
                    JOIN music m ON ph.music_id = m.id
                    WHERE ph.user_id = %s
                    ORDER BY ph.play_time DESC
                """, (user_id,))

                play_history = cursor.fetchall()

                if not play_history:
                    # 如果没有播放历史，使用基于评论的推荐
                    return self._get_recommendations_by_comments(user_id, limit)

                # 2. 获取相似用户的播放历史
                similar_users = self._get_similar_users(user_id, play_history, cursor)
                
                if not similar_users:
                    return self._get_recommendations_by_comments(user_id, limit)

                # 3. 获取推荐音乐
                recommendations = self._get_collaborative_recommendations(
                    user_id, similar_users, play_history, cursor, limit
                )

                if not recommendations:
                    return self._get_recommendations_by_comments(user_id, limit)

                return recommendations

            except Exception as e:
                print(f"推荐生成错误: {str(e)}")
                return self._get_popular_recommendations(cursor) if 'cursor' in locals() else []

            finally:
                if 'cursor' in locals():
                    cursor.close()
                if 'conn' in locals():
                    conn.close()

    def _get_similar_users(self, user_id, play_history, cursor):
        """获取相似用户"""
        try:
            # 获取当前用户的音乐ID列表
            user_music_ids = [record[0] for record in play_history]
            
            if not user_music_ids:
                return []

            # 查找听过相同音乐的用户
            placeholders = ','.join(['%s'] * len(user_music_ids))
            cursor.execute(f"""
                SELECT 
                    ph.user_id,
                    COUNT(DISTINCT ph.music_id) as common_music_count,
                    MAX(ph.play_time) as last_play_time
                FROM play_history ph
                WHERE ph.user_id != %s
                AND ph.music_id IN ({placeholders})
                GROUP BY ph.user_id
                HAVING common_music_count >= 2
                ORDER BY common_music_count DESC, last_play_time DESC
                LIMIT 10
            """, [user_id] + user_music_ids)

            return cursor.fetchall()
        except Exception as e:
            print(f"获取相似用户失败: {str(e)}")
            return []

    def _get_collaborative_recommendations(self, user_id, similar_users, play_history, cursor, limit):
        """获取协同过滤推荐"""
        try:
            # 获取当前用户已播放的音乐ID
            user_played_ids = {record[0] for record in play_history}
            
            # 获取相似用户ID列表
            similar_user_ids = [user[0] for user in similar_users]
            
            if not similar_user_ids:
                return []

            # 获取相似用户播放过的音乐
            placeholders = ','.join(['%s'] * len(similar_user_ids))
            cursor.execute(f"""
                SELECT 
                    m.id,
                    m.music_name,
                    m.singer,
                    m.languages,
                    m.school,
                    m.image_url,
                    COUNT(DISTINCT ph.user_id) as user_count,
                    AVG(TIMESTAMPDIFF(HOUR, ph.play_time, NOW())) as avg_hours_ago
                FROM play_history ph
                JOIN music m ON ph.music_id = m.id
                WHERE ph.user_id IN ({placeholders})
                AND ph.music_id NOT IN (
                    SELECT music_id 
                    FROM play_history 
                    WHERE user_id = %s
                )
                GROUP BY m.id, m.music_name, m.singer, m.languages, m.school, m.image_url
                HAVING user_count >= 2
                ORDER BY user_count DESC, avg_hours_ago ASC
                LIMIT %s
            """, similar_user_ids + [user_id, limit * 2])

            candidates = cursor.fetchall()

            if not candidates:
                return []

            # 计算推荐分数
            recommendations = []
            for music in candidates:
                # 基础分数：听过该音乐的用户数
                base_score = music[6]
                
                # 时间衰减因子：越近期的播放权重越大
                time_factor = 1.0 / (1.0 + music[7] / 24)  # 转换为天
                
                # 最终分数
                final_score = base_score * time_factor
                
                recommendations.append({
                    'id': music[0],
                    'music_name': music[1],
                    'singer': music[2],
                    'languages': music[3] or '其他',
                    'school': music[4] or '其他',
                    'image_url': music[5],
                    'score': round(final_score * 10, 1)  # 转换为0-100的分数
                })

            # 按分数排序并返回
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            return recommendations[:limit]

        except Exception as e:
            print(f"获取协同过滤推荐失败: {str(e)}")
            return []

    def _get_recommendations_by_comments(self, user_id, limit):
        """基于评论的推荐"""
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # 获取用户的评论历史
            cursor.execute("""
                SELECT 
                    m.id,
                    m.music_name,
                    m.singer,
                    m.languages,
                    m.school,
                    mc.sentiment,
                    mc.like_count,
                    mc.comment_time
                FROM music_comments mc
                JOIN music m ON mc.music_id = m.id
                WHERE mc.user_id = %s
                ORDER BY mc.comment_time DESC
            """, (user_id,))

            user_history = cursor.fetchall()

            if not user_history:
                return self._get_popular_recommendations(cursor)

            # 分析用户偏好
            preferences = self._analyze_user_preferences(user_history, [])

            # 获取候选音乐
            cursor.execute("""
                SELECT DISTINCT
                    m.id,
                    m.music_name,
                    m.singer,
                    m.languages,
                    m.school,
                    m.image_url,
                    m.play_count,
                    m.like_count,
                    m.comment_count
                FROM music m
                WHERE m.id NOT IN (
                    SELECT DISTINCT music_id 
                    FROM music_comments 
                    WHERE user_id = %s
                )
            """, (user_id,))

            candidates = cursor.fetchall()

            if not candidates:
                return self._get_popular_recommendations(cursor)

            # 计算推荐分数
            recommendations = []
            for music in candidates:
                score = self._calculate_recommendation_score(music, preferences)
                recommendations.append({
                    'id': music[0],
                    'music_name': music[1],
                    'singer': music[2],
                    'languages': music[3] or '其他',
                    'school': music[4] or '其他',
                    'image_url': music[5],
                    'score': score
                })

            # 排序并返回推荐结果
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            return recommendations[:limit]

        except Exception as e:
            print(f"基于评论的推荐失败: {str(e)}")
            return self._get_popular_recommendations(cursor) if 'cursor' in locals() else []

        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()

    def _analyze_user_preferences(self, history, play_history):
        """分析用户偏好"""
        preferences = {
            'languages': defaultdict(float),
            'school': defaultdict(float),
            'singers': defaultdict(float),
            'liked_music_ids': set(),
            'recent_weight': 1.0
        }

        total_interactions = len(history) + len(play_history)
        if total_interactions == 0:
            return preferences

        # 计算时间衰减权重
        latest_time = max(
            max((row[7] for row in history), default=None),
            max((row[5] for row in play_history), default=None)
        )

        # 处理评论历史
        for i, record in enumerate(history):
            # 基础权重（越近期的评论权重越大）
            time_weight = 1.0 - (0.8 * i / total_interactions)

            # 情感权重
            sentiment_weight = 1.5 if record[5] == '积极' else 0.5

            # 点赞权重
            like_weight = min(record[6] / 10.0, 1.0) if record[6] else 0.0

            # 综合权重
            weight = time_weight * (1 + sentiment_weight + like_weight)

            # 更新各维度偏好
            if record[3]:  # languages
                preferences['languages'][record[3]] += weight
            if record[4]:  # school
                preferences['school'][record[4]] += weight
            if record[2]:  # singer
                preferences['singers'][record[2]] += weight

            # 记录用户喜欢的音乐
            if record[5] == '积极':
                preferences['liked_music_ids'].add(record[0])

        # 处理播放历史
        for i, record in enumerate(play_history):
            # 播放历史权重（比评论权重略低）
            time_weight = 0.8 * (1.0 - (0.8 * i / total_interactions))
            
            # 更新各维度偏好（增加播放历史的权重）
            if record[3]:  # languages
                preferences['languages'][record[3]] += time_weight * 1.5  # 增加权重
            if record[4]:  # school
                preferences['school'][record[4]] += time_weight * 1.5
            if record[2]:  # singer
                preferences['singers'][record[2]] += time_weight * 1.5

        # 归一化
        for category in ['languages', 'school', 'singers']:
            max_val = max(preferences[category].values()) if preferences[category] else 1
            for key in preferences[category]:
                preferences[category][key] /= max_val

        return preferences

    def _calculate_recommendation_score(self, music, preferences):
        """计算推荐分数"""
        try:
            base_score = 0

            # 1. 语种匹配度 (30%)
            if music[3] in preferences['languages']:
                base_score += 30 * preferences['languages'][music[3]]

            # 2. 流派匹配度 (30%)
            if music[4] in preferences['school']:
                base_score += 30 * preferences['school'][music[4]]

            # 3. 歌手匹配度 (40%)
            if music[2] in preferences['singers']:
                base_score += 40 * preferences['singers'][music[2]]

            # 4. 热度影响 (额外加分)
            popularity_score = (
                                       (music[6] or 0) * 0.4 +  # play_count
                                       (music[7] or 0) * 0.3 +  # like_count
                                       (music[8] or 0) * 0.3  # comment_count
                               ) / 1000  # 归一化

            # 最终分数：基础分数 + 热度加成（最多10分）
            final_score = base_score + min(popularity_score, 10)

            # 确保分数在0-100之间
            return round(min(max(final_score, 0), 100), 1)

        except Exception as e:
            print(f"计算推荐分数错误: {str(e)}")
            return 0

    def _get_popular_recommendations(self, cursor, limit=12):
        """获取热门音乐推荐"""
        try:
            cursor.execute("""
                SELECT 
                    id, music_name, singer, languages, school, image_url,
                    ROUND((play_count * 0.4 + like_count * 0.3 + comment_count * 0.3) / 
                          GREATEST(
                              (SELECT MAX(play_count * 0.4 + like_count * 0.3 + comment_count * 0.3) FROM music),
                              1
                          ) * 100) as score
                FROM music
                ORDER BY score DESC
                LIMIT %s
            """, (limit,))

            recommendations = cursor.fetchall()
            return self._format_recommendations(recommendations)
        except Exception as e:
            print(f"获取热门推荐失败: {str(e)}")
            return []

    def _format_recommendations(self, recommendations):
        """格式化推荐结果"""
        result = []
        for rec in recommendations:
            result.append({
                'id': rec[0],
                'music_name': rec[1],
                'singer': rec[2],
                'languages': rec[3] or '其他',
                'school': rec[4] or '其他',
                'image_url': rec[5],
                'score': round(float(rec[6]), 1)
            })
        return result

    def get_associated_recommendations(self, music_id, limit=12):
        """获取音乐关联推荐"""
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # 1. 基于共同用户行为的关联分析
            cursor.execute("""
                WITH user_interactions AS (
                    SELECT DISTINCT user_id 
                    FROM music_comments 
                    WHERE music_id = %s
                )
                SELECT 
                    m.id,
                    m.music_name,
                    m.singer,
                    m.languages,
                    m.school,
                    m.image_url,
                    COUNT(DISTINCT mc.user_id) as common_users,
                    SUM(CASE WHEN mc.sentiment = '积极' THEN 1 ELSE 0 END) as positive_count
                FROM music m
                JOIN music_comments mc ON m.id = mc.music_id
                WHERE mc.user_id IN (SELECT user_id FROM user_interactions)
                AND m.id != %s
                GROUP BY m.id, m.music_name, m.singer, m.languages, m.school, m.image_url
                HAVING common_users >= 2
            """, (music_id, music_id))

            associated_items = cursor.fetchall()

            # 2. 获取当前音乐的特征
            cursor.execute("""
                SELECT languages, school, singer
                FROM music
                WHERE id = %s
            """, (music_id,))
            current_music = cursor.fetchone()

            # 3. 计算关联分数
            recommendations = []
            for item in associated_items:
                base_score = 0
                # 共同用户数量权重 (40%)
                user_weight = min(item[6] / 10.0, 1.0) * 40

                # 正面评价比例权重 (30%)
                sentiment_weight = (item[7] / item[6]) * 30 if item[6] > 0 else 0

                # 特征相似度权重 (30%)
                feature_weight = 0
                if item[3] == current_music[0]:  # 相同语种
                    feature_weight += 10
                if item[4] == current_music[1]:  # 相同流派
                    feature_weight += 10
                if item[2] == current_music[2]:  # 相同歌手
                    feature_weight += 10

                total_score = user_weight + sentiment_weight + feature_weight

                recommendations.append({
                    'id': item[0],
                    'music_name': item[1],
                    'singer': item[2],
                    'languages': item[3] or '其他',
                    'school': item[4] or '其他',
                    'image_url': item[5],
                    'score': round(total_score, 1)
                })

            # 按分数排序
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            return recommendations[:limit]

        except Exception as e:
            print(f"获取关联推荐失败: {str(e)}")
            return []

        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'spark') and self.spark:
            try:
                self.spark.stop()
            except:
                pass

    def get_popular_music_by_type(self, type_filter=None, time_range='all', limit=12):
        """
        获取流行音乐推荐

        参数:
            type_filter: 筛选类型，格式为 'field:value'，如 'languages:华语' 或 'school:流行'
            time_range: 时间范围 ('week'/'month'/'year'/'all')
            limit: 返回数量
        """
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # 构建时间范围条件
            time_condition = ""
            if time_range == 'week':
                time_condition = "AND mc.comment_time >= DATE_SUB(NOW(), INTERVAL 7 DAY)"
            elif time_range == 'month':
                time_condition = "AND mc.comment_time >= DATE_SUB(NOW(), INTERVAL 30 DAY)"
            elif time_range == 'year':
                time_condition = "AND mc.comment_time >= DATE_SUB(NOW(), INTERVAL 1 YEAR)"

            # 构建类型筛选条件
            type_condition = ""
            if type_filter:
                field, value = type_filter.split(':')
                type_condition = f"AND m.{field} = %s"

            query = """
                SELECT 
                    m.id,
                    m.music_name,
                    m.singer,
                    m.languages,
                    m.school,
                    m.image_url,
                    m.play_count,
                    m.like_count,
                    m.comment_count,
                    COUNT(DISTINCT mc.user_id) as listener_count,
                    SUM(CASE WHEN mc.sentiment = '积极' THEN 1 ELSE 0 END) as positive_count
                FROM music m
                LEFT JOIN music_comments mc ON m.id = mc.music_id
                WHERE 1=1 
                """ + time_condition + type_condition + """
                GROUP BY m.id
                ORDER BY (
                    COALESCE(m.play_count, 0) * 0.3 + 
                    COALESCE(m.like_count, 0) * 0.3 + 
                    COALESCE(m.comment_count, 0) * 0.2 +
                    COALESCE(COUNT(DISTINCT mc.user_id), 0) * 0.2
                ) DESC
                LIMIT %s
            """

            params = []
            if type_filter:
                params.append(type_filter.split(':')[1])
            params.append(limit)

            cursor.execute(query, tuple(params))
            results = cursor.fetchall()

            # 格式化结果
            popular_music = []
            for row in results:
                total_score = (
                        (row[6] or 0) * 0.3 +  # play_count
                        (row[7] or 0) * 0.3 +  # like_count
                        (row[8] or 0) * 0.2 +  # comment_count
                        (row[9] or 0) * 0.2  # listener_count
                )

                popular_music.append({
                    'id': row[0],
                    'music_name': row[1],
                    'singer': row[2],
                    'languages': row[3] or '其他',
                    'school': row[4] or '其他',
                    'image_url': row[5],
                    'play_count': row[6] or 0,
                    'like_count': row[7] or 0,
                    'comment_count': row[8] or 0,
                    'listener_count': row[9] or 0,
                    'positive_rate': round((row[10] or 0) / (row[9] or 1) * 100, 1),
                    'score': round(total_score / 1000, 1)  # 归一化分数
                })

            return popular_music

        except Exception as e:
            print(f"获取流行音乐失败: {str(e)}")
            return []

        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()

    def get_music_categories(self):
        """获取音乐分类信息"""
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # 获取语种分类
            cursor.execute("""
                SELECT DISTINCT languages, COUNT(*) as count
                FROM music 
                WHERE languages IS NOT NULL
                GROUP BY languages
                ORDER BY count DESC
            """)
            languages = [{'value': row[0], 'count': row[1]} for row in cursor.fetchall()]

            # 获取流派分类
            cursor.execute("""
                SELECT DISTINCT school, COUNT(*) as count
                FROM music 
                WHERE school IS NOT NULL
                GROUP BY school
                ORDER BY count DESC
            """)
            schools = [{'value': row[0], 'count': row[1]} for row in cursor.fetchall()]

            return {
                'languages': languages,
                'school': schools
            }

        except Exception as e:
            print(f"获取分类信息失败: {str(e)}")
            return {'languages': [], 'school': []}

        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()


class MusicRecommendationMRJob(MRJob):
    """音乐推荐MapReduce作业类"""
    
    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_ratings,
                  reducer=self.reducer_sum_ratings),
            MRStep(reducer=self.reducer_calculate_similarities)
        ]
    
    def mapper_get_ratings(self, _, line):
        """映射用户评分数据"""
        try:
            data = json.loads(line)
            user_id = data['user_id']
            music_id = data['music_id']
            rating = data['rating']
            yield music_id, (user_id, rating)
        except:
            pass
            
    def reducer_sum_ratings(self, music_id, user_ratings):
        """汇总音乐评分"""
        ratings = {}
        for user_id, rating in user_ratings:
            ratings[user_id] = rating
        yield music_id, ratings
        
    def reducer_calculate_similarities(self, music_id, ratings_list):
        """计算音乐相似度"""
        all_ratings = {}
        for ratings in ratings_list:
            all_ratings.update(ratings)
            
        similarities = self._calculate_similarity(all_ratings)
        yield music_id, similarities
        
    def _calculate_similarity(self, ratings):
        """计算余弦相似度"""
        return ratings  # 简化版实现


class HadoopRecommendationService:
    """基于Hadoop的音乐推荐服务"""
    
    def __init__(self, hdfs_host='localhost', hdfs_port=9000, 
                 namenode='localhost:9000'):
        """初始化Hadoop推荐服务"""
        self.hdfs_client = InsecureClient(f'http://{hdfs_host}:{hdfs_port}')
        self.namenode = namenode
        self.hdfs_music_data = '/music/data'
        self.hdfs_user_data = '/music/users'
        self.hdfs_recommendation_output = '/music/recommendations'
        
        # 确保HDFS目录存在
        self._ensure_hdfs_dirs()
        
    def _ensure_hdfs_dirs(self):
        """确保HDFS所需目录存在"""
        for path in [self.hdfs_music_data, self.hdfs_user_data, 
                    self.hdfs_recommendation_output]:
            try:
                self.hdfs_client.makedirs(path)
            except:
                pass
                
    def upload_music_data(self, music_data: List[Dict[str, Any]]):
        """上传音乐数据到HDFS"""
        try:
            with self.hdfs_client.write(f'{self.hdfs_music_data}/music.json', 
                                      encoding='utf-8') as writer:
                for item in music_data:
                    writer.write(json.dumps(item, ensure_ascii=False) + '\n')
            return True
        except Exception as e:
            print(f"上传音乐数据失败: {str(e)}")
            return False
            
    def upload_user_data(self, user_data: List[Dict[str, Any]]):
        """上传用户数据到HDFS"""
        try:
            with self.hdfs_client.write(f'{self.hdfs_user_data}/users.json',
                                      encoding='utf-8') as writer:
                for item in user_data:
                    writer.write(json.dumps(item, ensure_ascii=False) + '\n')
            return True
        except Exception as e:
            print(f"上传用户数据失败: {str(e)}")
            return False
            
    def run_recommendation_job(self) -> Dict[str, List[str]]:
        """运行推荐MapReduce作业"""
        try:
            # 配置MRJob
            mr_job = MusicRecommendationMRJob(args=[
                '-r', 'hadoop',
                '--hadoop-streaming-jar', 
                '/usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming.jar',
                f'{self.hdfs_music_data}/music.json'
            ])
            
            # 运行作业
            with mr_job.make_runner() as runner:
                runner.run()
                
                # 收集结果
                recommendations = {}
                for line in runner.stream_output():
                    music_id, similar_items = mr_job.parse_output_line(line)
                    recommendations[music_id] = similar_items
                    
                return recommendations
                
        except Exception as e:
            print(f"推荐作业运行失败: {str(e)}")
            return {}
            
    def get_hadoop_recommendations(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """获取基于Hadoop的推荐结果"""
        try:
            # 获取用户数据
            user_data = self._get_user_data(user_id)
            if not user_data:
                return []
                
            # 运行推荐作业
            recommendations = self.run_recommendation_job()
            
            # 处理推荐结果
            user_recs = self._process_recommendations(
                user_id, recommendations, user_data, limit
            )
            
            return user_recs
            
        except Exception as e:
            print(f"获取Hadoop推荐失败: {str(e)}")
            return []
            
    def _get_user_data(self, user_id: int) -> Dict[str, Any]:
        """从HDFS获取用户数据"""
        try:
            with self.hdfs_client.read(f'{self.hdfs_user_data}/users.json',
                                     encoding='utf-8') as reader:
                for line in reader:
                    data = json.loads(line)
                    if data.get('user_id') == user_id:
                        return data
            return {}
        except:
            return {}
            
    def _process_recommendations(
        self, user_id: int, 
        recommendations: Dict[str, List[str]],
        user_data: Dict[str, Any],
        limit: int
    ) -> List[Dict[str, Any]]:
        """处理推荐结果"""
        try:
            # 获取用户历史记录
            user_history = set(user_data.get('history', []))
            
            # 过滤和排序推荐
            filtered_recs = []
            for music_id, similar_items in recommendations.items():
                if music_id not in user_history:
                    score = self._calculate_recommendation_score(
                        music_id, similar_items, user_data
                    )
                    filtered_recs.append({
                        'music_id': music_id,
                        'score': score
                    })
                    
            # 排序并限制数量
            filtered_recs.sort(key=lambda x: x['score'], reverse=True)
            return filtered_recs[:limit]
            
        except Exception as e:
            print(f"处理推荐结果失败: {str(e)}")
            return []
            
    def _calculate_recommendation_score(
        self, music_id: str,
        similar_items: List[str],
        user_data: Dict[str, Any]
    ) -> float:
        """计算推荐分数"""
        try:
            # 基础分数计算
            base_score = len(set(similar_items) & 
                           set(user_data.get('liked_items', [])))
            
            # 用户偏好权重
            preferences = user_data.get('preferences', {})
            preference_score = sum(
                preferences.get(pref, 0) for pref in 
                self._get_music_preferences(music_id)
            )
            
            # 综合评分
            return base_score * 0.7 + preference_score * 0.3
            
        except:
            return 0.0
            
    def _get_music_preferences(self, music_id: str) -> List[str]:
        """获取音乐特征"""
        try:
            with self.hdfs_client.read(f'{self.hdfs_music_data}/music.json',
                                     encoding='utf-8') as reader:
                for line in reader:
                    data = json.loads(line)
                    if data.get('id') == music_id:
                        return [
                            data.get('languages', ''),
                            data.get('school', ''),
                            data.get('singer', '')
                        ]
            return []
        except:
            return []