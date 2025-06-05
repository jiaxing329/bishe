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

    def get_recommendations(self, user_id, limit=12):
        """获取音乐推荐"""
        # 如果启用了Spark且初始化成功，使用Spark推荐
        if self.use_spark:
            try:
                return self.get_spark_recommendations(user_id, limit)
            except Exception as e:
                print(f"Spark推荐失败，切换到SQL推荐: {str(e)}")
        
        # 使用SQL推荐（保持原有代码不变）
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            # 1. 获取用户的评论历史和情感倾向
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
                
            # 2. 分析用户偏好
            preferences = self._analyze_user_preferences(user_history)
            
            # 3. 获取候选音乐
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
            
            # 4. 计算推荐分数
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
            
            # 5. 排序并返回推荐结果
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            return recommendations[:limit]
            
        except Exception as e:
            print(f"推荐生成错误: {str(e)}")
            return self._get_popular_recommendations(cursor) if 'cursor' in locals() else []
            
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()
    
    def _analyze_user_preferences(self, history):
        """分析用户偏好"""
        preferences = {
            'languages': defaultdict(float),
            'school': defaultdict(float),
            'singers': defaultdict(float),
            'liked_music_ids': set(),
            'recent_weight': 1.0
        }
        
        total_interactions = len(history)
        if total_interactions == 0:
            return preferences
            
        # 计算时间衰减权重
        latest_time = max(row[7] for row in history)
        
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
                (music[8] or 0) * 0.3    # comment_count
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

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'spark') and self.spark:
            try:
                self.spark.stop()
            except:
                pass 