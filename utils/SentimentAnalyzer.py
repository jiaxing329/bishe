from snownlp import SnowNLP
import re

class SentimentAnalyzer:
    # 扩展情感词典
    negative_words = {
        '难听', '垃圾', '差劲', '糟糕', '难受', '讨厌', '恶心', '烂', '差', '废物',
        '失望', '无聊', '乏味', '单调', '枯燥', '噪音', '刺耳', '难以接受',
        '浪费', '后悔', '无感', '不堪入耳', '低级', '廉价', '敷衍', '马虎',
        '过时', '过气', '俗气', '土气', '不专业', '业余', '粗糙', '简陋'
    }
    
    positive_words = {
        '好听', '优秀', '棒', '赞', '喜欢', '不错', '牛', '强', '厉害', '精彩',
        '感动', '震撼', '惊艳', '完美', '经典', '大师', '天籁', '享受',
        '专业', '细腻', '动听', '悦耳', '温暖', '治愈', '舒服', '优雅',
        '高级', '大气', '精致', '用心', '走心', '真诚', '深情', '动人'
    }
    
    # 情感强度词
    intensity_words = {
        '非常': 2.0, '很': 1.5, '特别': 1.8, '真': 1.5, '太': 1.8,
        '超': 1.8, '好': 1.5, '极其': 2.0, '格外': 1.8, '尤其': 1.5
    }
    
    @staticmethod
    def analyze(text):
        """
        分析文本情感倾向
        返回: 情感分析结果
        """
        try:
            if not text:
                return '中性'
                
            # 文本预处理
            text = text.lower()  # 转小写
            
            # 基础情感分数
            s = SnowNLP(text)
            score = s.sentiments
            
            # 计算情感词出现次数和强度
            negative_count = 0
            positive_count = 0
            total_intensity = 1.0
            
            # 检查情感强度词
            for intensity_word, multiplier in SentimentAnalyzer.intensity_words.items():
                if intensity_word in text:
                    total_intensity *= multiplier
            
            # 统计情感词
            for word in SentimentAnalyzer.negative_words:
                if word in text:
                    negative_count += 1
                    
            for word in SentimentAnalyzer.positive_words:
                if word in text:
                    positive_count += 1
            
            # 检查否定词
            negation_words = {'不', '没', '不是', '不能', '不可以', '不行'}
            has_negation = any(word in text for word in negation_words)
            
            # 调整分数
            if negative_count > 0:
                score *= (1 - negative_count * 0.2)
            if positive_count > 0:
                score = min(1.0, score * (1 + positive_count * 0.2))
            
            # 应用情感强度
            score *= total_intensity
            
            # 处理否定词
            if has_negation:
                score = 1 - score
            
            # 返回情感判断结果
            if score >= 0.8:
                return '非常积极'
            elif score >= 0.6:
                return '积极'
            elif score >= 0.45 and score < 0.55:  # 缩小中性范围
                return '中性'
            elif score >= 0.2:
                return '消极'
            else:
                return '非常消极'
            
        except Exception as e:
            print(f"情感分析失败: {str(e)}")
            return '中性'