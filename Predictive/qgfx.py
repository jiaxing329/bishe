import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import jieba
import pymysql  # 改用 pymysql
from sklearn.metrics import accuracy_score, classification_report
import warnings
import traceback
warnings.filterwarnings('ignore')

class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = LogisticRegression(random_state=42)
        # 简化数据库配置
        self.db_config = {
            'host': '127.0.0.1',
            'port': 3306,
            'user': 'root',
            'password': '123456',
            'charset': 'utf8mb4',
            'cursorclass': pymysql.cursors.DictCursor
        }
        
        # 测试数据库连接
        self.test_database_connection()

    def test_database_connection(self):
        """测试数据库连接"""
        print("测试数据库连接...")
        try:
            print("尝试连接到MySQL服务器...")
            conn = pymysql.connect(**self.db_config)
            cursor = conn.cursor()
            
            # 测试服务器连接
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()
            print(f"成功连接到MySQL服务器（版本：{version['VERSION()']}）")
            
            # 检查数据库是否存在
            cursor.execute("SHOW DATABASES")
            databases = [db['Database'] for db in cursor.fetchall()]
            if 'py_music' not in databases:
                print(f"数据库 py_music 不存在")
                print("现有数据库:", databases)
                raise Exception("数据库 py_music 不存在")
            
            # 切换到指定数据库
            cursor.execute("USE py_music")
            
            # 检查表是否存在
            cursor.execute("SHOW TABLES")
            tables = [table['Tables_in_py_music'] for table in cursor.fetchall()]
            if 'music_comments' not in tables:
                print("music_comments 表不存在")
                print("现有表:", tables)
                raise Exception("music_comments 表不存在")
            
            # 检查表结构
            cursor.execute("DESCRIBE music_comments")
            columns = cursor.fetchall()
            print("\n表结构:")
            for col in columns:
                print(f"- {col['Field']}: {col['Type']}")
            
            print("\n数据库连接测试成功")
            
        except pymysql.Error as err:
            print(f"MySQL错误: {err}")
            raise
        except Exception as e:
            print(f"测试连接时发生错误: {e}")
            raise
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()

    def preprocess_text(self, text):
        """文本预处理"""
        if isinstance(text, str):
            # 分词
            words = jieba.cut(text)
            # 转换为空格分隔的字符串
            return ' '.join(words)
        return ''

    def train_model(self):
        """训练模型"""
        print("开始训练模型...")
        
        try:
            # 读取训练数据
            print("正在读取训练数据...")
            df = None
            error_messages = []
            
            try:
                df = pd.read_csv('./all.csv', encoding='gbk')
                print("从当前目录成功读取数据")
            except Exception as e:
                error_messages.append(f"尝试读取 ./all.csv (gbk) 失败: {str(e)}")
                try:
                    df = pd.read_csv('./all.csv', encoding='utf-8')
                    print("从当前目录成功读取数据 (utf-8)")
                except Exception as e:
                    error_messages.append(f"尝试读取 ./all.csv (utf-8) 失败: {str(e)}")
                    try:
                        df = pd.read_csv('./Predictive/all.csv', encoding='gbk')
                        print("从 Predictive 目录成功读取数据")
                    except Exception as e:
                        error_messages.append(f"尝试读取 ./Predictive/all.csv 失败: {str(e)}")
            
            if df is None:
                raise Exception(f"无法读取训练数据文件。错误信息：\n" + "\n".join(error_messages))
            
            print(f"成功读取训练数据，共 {len(df)} 条记录")
            print("数据列名:", df.columns.tolist())
            
            # 打印数据样本，帮助调试
            print("\n数据样本（前5行）:")
            print(df.head())
            print("\n标签值统计:")
            print(df['label'].value_counts())
            
            # 使用实际的列名
            comment_col = 'evaluation'  # 评论内容列
            sentiment_col = 'label'     # 情感标签列
            
            print(f"\n使用列名 - 评论内容: {comment_col}, 情感标签: {sentiment_col}")
            
            # 预处理文本
            print("开始预处理文本...")
            df['processed_text'] = df[comment_col].apply(self.preprocess_text)
            
            # 将情感标签转换为数值
            sentiment_map = {'积极': 1, '消极': 0}
            df['sentiment_label'] = df[sentiment_col].map(sentiment_map)
            
            # 检查数据有效性
            valid_mask = df['sentiment_label'].notna()
            invalid_count = (~valid_mask).sum()
            if invalid_count > 0:
                print(f"\n警告：发现 {invalid_count} 条无效的情感标签")
                print("无效标签值:", df[df['sentiment_label'].isna()][sentiment_col].unique())
            
            df = df[valid_mask]
            print(f"有效训练数据 {len(df)} 条")
            
            if len(df) == 0:
                raise Exception("没有有效的训练数据")
            
            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                df['processed_text'], 
                df['sentiment_label'],
                test_size=0.2,
                random_state=42
            )
            
            # 特征提取
            print("开始特征提取...")
            X_train_tfidf = self.vectorizer.fit_transform(X_train)
            X_test_tfidf = self.vectorizer.transform(X_test)
            
            # 训练模型
            print("开始训练模型...")
            self.model.fit(X_train_tfidf, y_train)
            
            # 评估模型
            y_pred = self.model.predict(X_test_tfidf)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"模型准确率: {accuracy:.4f}")
            print("\n分类报告:")
            print(classification_report(y_test, y_pred, target_names=['消极', '积极']))
            
        except Exception as e:
            print(f"训练模型时发生错误: {str(e)}")
            print("\n详细错误信息:")
            print(traceback.format_exc())
            raise

    def analyze_database_comments(self):
        """分析数据库中的评论"""
        print("开始分析数据库评论...")
        
        try:
            # 连接数据库
            conn = pymysql.connect(**self.db_config, db='py_music')
            cursor = conn.cursor()
            
            # 检查是否存在sentiment列
            cursor.execute("""
                SELECT COUNT(*)
                FROM information_schema.columns 
                WHERE table_schema = 'py_music'
                AND table_name = 'music_comments' 
                AND column_name = 'sentiment'
            """)
            
            if cursor.fetchone()['COUNT(*)'] == 0:
                # 添加sentiment列
                cursor.execute("""
                    ALTER TABLE music_comments 
                    ADD COLUMN sentiment VARCHAR(10) DEFAULT NULL
                """)
                conn.commit()
            
            # 获取所有未分析的评论
            cursor.execute("""
                SELECT id, comment_content 
                FROM music_comments 
                WHERE sentiment IS NULL 
                AND comment_content IS NOT NULL
                AND LENGTH(TRIM(comment_content)) > 0
            """)
            comments = cursor.fetchall()
            
            if not comments:
                print("没有需要分析的新评论")
                return
                
            print(f"找到 {len(comments)} 条需要分析的评论")
            
            # 批量处理评论
            batch_size = 1000
            for i in range(0, len(comments), batch_size):
                batch = comments[i:i + batch_size]
                
                # 预处理文本
                processed_texts = [self.preprocess_text(comment['comment_content']) for comment in batch]
                
                # 转换为TF-IDF特征
                X = self.vectorizer.transform(processed_texts)
                
                # 预测情感
                predictions = self.model.predict(X)
                
                # 更新数据库
                update_data = []
                for j, pred in enumerate(predictions):
                    sentiment = '积极' if pred == 1 else '消极'
                    update_data.append((sentiment, batch[j]['id']))
                
                # 批量更新
                cursor.executemany(
                    "UPDATE music_comments SET sentiment = %s WHERE id = %s",
                    update_data
                )
                conn.commit()
                
                print(f"已处理 {i + len(batch)} 条评论")
            
            print("评论情感分析完成")
            
        except pymysql.Error as err:
            print(f"数据库错误: {err}")
            print("\n详细错误信息:")
            print(traceback.format_exc())
        except Exception as e:
            print(f"发生错误: {str(e)}")
            print("\n详细错误信息:")
            print(traceback.format_exc())
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()

def main():
    try:
        print("正在初始化情感分析器...")
        analyzer = SentimentAnalyzer()
        print("初始化完成，开始训练模型...")
        # 训练模型
        analyzer.train_model()
        print("模型训练完成，开始分析评论...")
        # 分析数据库评论
        analyzer.analyze_database_comments()
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        print("\n详细错误信息:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
