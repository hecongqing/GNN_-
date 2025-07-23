import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import jieba
from collections import Counter
from typing import Dict, List, Tuple, Optional
import pickle
import re


class SentimentDataset(Dataset):
    """情感分析数据集"""
    
    def __init__(self, texts: List[str], labels: List[int], vocab: Dict[str, int], 
                 max_length: int = 128, tokenizer_type: str = 'jieba'):
        """
        初始化情感分析数据集
        
        Args:
            texts: 文本列表
            labels: 标签列表
            vocab: 词汇表
            max_length: 最大序列长度
            tokenizer_type: 分词器类型
        """
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        self.tokenizer_type = tokenizer_type
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 文本预处理和tokenization
        tokens = self.tokenize(text)
        
        # 转换为索引
        token_ids = [self.vocab.get(token, self.vocab.get('<UNK>', 1)) for token in tokens]
        
        # 截断或填充
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
            length = self.max_length
        else:
            length = len(token_ids)
            token_ids.extend([0] * (self.max_length - len(token_ids)))  # 0是<PAD>的索引
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long),
            'length': torch.tensor(length, dtype=torch.long)
        }
    
    def tokenize(self, text: str) -> List[str]:
        """文本分词"""
        # 清理文本
        text = self.clean_text(text)
        
        if self.tokenizer_type == 'jieba':
            # 使用jieba分词
            tokens = list(jieba.cut(text))
        else:
            # 简单的字符级分词
            tokens = list(text)
        
        # 过滤空字符
        tokens = [token.strip() for token in tokens if token.strip()]
        
        return tokens
    
    def clean_text(self, text: str) -> str:
        """清理文本"""
        # 去除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 去除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 去除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 去除首尾空格
        text = text.strip()
        
        return text


class SentimentDataProcessor:
    """情感分析数据处理器"""
    
    def __init__(self, data_dir: str = 'dataset', max_vocab_size: int = 10000,
                 max_length: int = 128, min_freq: int = 2):
        """
        初始化数据处理器
        
        Args:
            data_dir: 数据目录
            max_vocab_size: 最大词汇表大小
            max_length: 最大序列长度
            min_freq: 词汇最小频率
        """
        self.data_dir = data_dir
        self.max_vocab_size = max_vocab_size
        self.max_length = max_length
        self.min_freq = min_freq
        
        self.vocab = None
        self.label_encoder = None
        
    def load_data(self, filename: str = None) -> Tuple[List[str], List[str]]:
        """
        加载数据
        
        Args:
            filename: 数据文件名
            
        Returns:
            texts: 文本列表
            labels: 标签列表
        """
        if filename:
            file_path = os.path.join(self.data_dir, filename)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                texts = df['text'].tolist()
                labels = df['label'].tolist()
                return texts, labels
        
        # 如果没有真实数据，生成示例数据
        print("未找到数据文件，生成示例数据...")
        return self.generate_sample_data()
    
    def generate_sample_data(self, n_samples: int = 1000) -> Tuple[List[str], List[str]]:
        """
        生成示例数据
        
        Args:
            n_samples: 样本数量
            
        Returns:
            texts: 文本列表
            labels: 标签列表
        """
        # 正面情感示例
        positive_samples = [
            "这部电影真的很好看，强烈推荐！",
            "服务态度非常好，很满意。",
            "质量很不错，物超所值。",
            "今天心情特别好，阳光明媚。",
            "这家餐厅的菜品很美味。",
            "工作顺利，同事都很友善。",
            "这个产品设计得很精美。",
            "学习新技能让我感到充实。",
            "旅行很愉快，风景优美。",
            "朋友们都很支持我的决定。"
        ]
        
        # 负面情感示例
        negative_samples = [
            "这部电影太无聊了，浪费时间。",
            "服务态度很差，不推荐。",
            "质量不好，很失望。",
            "今天心情很糟糕，诸事不顺。",
            "这家餐厅的菜品很难吃。",
            "工作压力大，同事关系紧张。",
            "这个产品设计得很粗糙。",
            "学习进度缓慢，感到焦虑。",
            "旅行遇到很多问题，很糟心。",
            "朋友们都不理解我的想法。"
        ]
        
        # 中性情感示例
        neutral_samples = [
            "这是一部普通的电影。",
            "服务态度一般。",
            "质量还可以。",
            "今天是平常的一天。",
            "这家餐厅的菜品一般。",
            "工作如常进行。",
            "这个产品设计还行。",
            "学习进度正常。",
            "旅行还算顺利。",
            "朋友们有不同的看法。"
        ]
        
        texts = []
        labels = []
        
        # 生成样本
        for i in range(n_samples):
            sample_type = i % 3
            if sample_type == 0:
                text = np.random.choice(positive_samples)
                label = "positive"
            elif sample_type == 1:
                text = np.random.choice(negative_samples)
                label = "negative"
            else:
                text = np.random.choice(neutral_samples)
                label = "neutral"
                
            texts.append(text)
            labels.append(label)
        
        return texts, labels
    
    def build_vocab(self, texts: List[str]) -> Dict[str, int]:
        """
        构建词汇表
        
        Args:
            texts: 文本列表
            
        Returns:
            vocab: 词汇表
        """
        # 统计词频
        word_counter = Counter()
        
        for text in texts:
            # 使用jieba分词
            tokens = list(jieba.cut(text))
            tokens = [token.strip() for token in tokens if token.strip()]
            word_counter.update(tokens)
        
        # 构建词汇表
        vocab = {'<PAD>': 0, '<UNK>': 1}
        
        # 按频率排序，选择最常见的词
        most_common = word_counter.most_common(self.max_vocab_size - 2)
        
        for word, freq in most_common:
            if freq >= self.min_freq:
                vocab[word] = len(vocab)
        
        print(f"词汇表大小: {len(vocab)}")
        return vocab
    
    def encode_labels(self, labels: List[str]) -> List[int]:
        """
        编码标签
        
        Args:
            labels: 标签列表
            
        Returns:
            encoded_labels: 编码后的标签
        """
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            encoded_labels = self.label_encoder.fit_transform(labels)
        else:
            encoded_labels = self.label_encoder.transform(labels)
        
        return encoded_labels.tolist()
    
    def prepare_data(self, test_size: float = 0.2, val_size: float = 0.1,
                    random_state: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        准备训练、验证和测试数据
        
        Args:
            test_size: 测试集比例
            val_size: 验证集比例
            random_state: 随机种子
            
        Returns:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
        """
        # 加载数据
        texts, labels = self.load_data()
        
        # 构建词汇表
        self.vocab = self.build_vocab(texts)
        
        # 编码标签
        encoded_labels = self.encode_labels(labels)
        
        # 分割数据
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, encoded_labels, test_size=test_size, random_state=random_state,
            stratify=encoded_labels
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state,
            stratify=y_temp
        )
        
        # 创建数据集
        train_dataset = SentimentDataset(X_train, y_train, self.vocab, self.max_length)
        val_dataset = SentimentDataset(X_val, y_val, self.vocab, self.max_length)
        test_dataset = SentimentDataset(X_test, y_test, self.vocab, self.max_length)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def save_vocab(self, filepath: str):
        """保存词汇表"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.vocab, f)
    
    def load_vocab(self, filepath: str):
        """加载词汇表"""
        with open(filepath, 'rb') as f:
            self.vocab = pickle.load(f)
    
    def save_label_encoder(self, filepath: str):
        """保存标签编码器"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.label_encoder, f)
    
    def load_label_encoder(self, filepath: str):
        """加载标签编码器"""
        with open(filepath, 'rb') as f:
            self.label_encoder = pickle.load(f)
    
    def get_num_classes(self) -> int:
        """获取类别数量"""
        if self.label_encoder is None:
            return 3  # 默认3类：正面、负面、中性
        return len(self.label_encoder.classes_)
    
    def get_vocab_size(self) -> int:
        """获取词汇表大小"""
        if self.vocab is None:
            return 0
        return len(self.vocab)