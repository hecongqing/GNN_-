import os
import torch
import logging
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import jieba
from wordcloud import WordCloud
from typing import Dict, List, Any, Optional


def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """设置日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 创建formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def set_random_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model: torch.nn.Module, filepath: str, epoch: int = None, 
               metric: float = None, additional_info: Dict = None):
    """保存模型"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'vocab_size': model.vocab_size,
            'embedding_dim': model.embedding_dim,
            'hidden_dim': model.hidden_dim,
            'num_layers': model.num_layers,
            'num_classes': model.num_classes
        }
    }
    
    if epoch is not None:
        save_dict['epoch'] = epoch
    if metric is not None:
        save_dict['metric'] = metric
    if additional_info:
        save_dict.update(additional_info)
    
    torch.save(save_dict, filepath)


def load_model(model: torch.nn.Module, filepath: str, device: str = 'cpu'):
    """加载模型"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint


def text_preprocess(text: str) -> str:
    """文本预处理"""
    # 转换为小写
    text = text.lower()
    
    # 去除特殊字符（保留中文、英文、数字）
    import re
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
    
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def tokenize_chinese(text: str) -> List[str]:
    """中文分词"""
    # 预处理
    text = text_preprocess(text)
    
    # 使用jieba分词
    tokens = list(jieba.cut(text))
    
    # 过滤空token
    tokens = [token.strip() for token in tokens if token.strip()]
    
    return tokens


def build_vocab_from_texts(texts: List[str], max_vocab_size: int = 10000, 
                          min_freq: int = 2) -> Dict[str, int]:
    """从文本构建词汇表"""
    from collections import Counter
    
    word_counter = Counter()
    
    for text in texts:
        tokens = tokenize_chinese(text)
        word_counter.update(tokens)
    
    # 构建词汇表
    vocab = {'<PAD>': 0, '<UNK>': 1}
    
    # 按频率排序，选择最常见的词
    most_common = word_counter.most_common(max_vocab_size - 2)
    
    for word, freq in most_common:
        if freq >= min_freq:
            vocab[word] = len(vocab)
    
    return vocab


def texts_to_sequences(texts: List[str], vocab: Dict[str, int], 
                      max_length: int = 128) -> List[List[int]]:
    """将文本转换为序列"""
    sequences = []
    
    for text in texts:
        tokens = tokenize_chinese(text)
        sequence = [vocab.get(token, vocab.get('<UNK>', 1)) for token in tokens]
        
        # 截断或填充
        if len(sequence) > max_length:
            sequence = sequence[:max_length]
        else:
            sequence.extend([0] * (max_length - len(sequence)))
        
        sequences.append(sequence)
    
    return sequences


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], 
                         class_names: List[str], save_path: str = None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_history(train_losses: List[float], val_losses: List[float],
                         train_accs: List[float], val_accs: List[float],
                         save_path: str = None):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def generate_wordcloud(texts: List[str], save_path: str = None):
    """生成词云"""
    # 合并所有文本
    all_text = ' '.join(texts)
    
    # 分词
    tokens = tokenize_chinese(all_text)
    text_for_wordcloud = ' '.join(tokens)
    
    # 生成词云
    wordcloud = WordCloud(
        font_path='SimHei.ttf',  # 需要中文字体
        width=800,
        height=400,
        background_color='white',
        max_words=100,
        colormap='viridis'
    ).generate(text_for_wordcloud)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def print_classification_report(y_true: List[int], y_pred: List[int], 
                               class_names: List[str] = None):
    """打印分类报告"""
    if class_names is None:
        class_names = [f'Class {i}' for i in range(max(max(y_true), max(y_pred)) + 1)]
    
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("Classification Report:")
    print("=" * 60)
    print(report)


def calculate_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """计算评估指标"""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, 
                 mode: str = 'min', verbose: bool = True):
        """
        Args:
            patience: 容忍的epoch数量
            min_delta: 最小改善阈值
            mode: 'min'表示监控指标越小越好，'max'表示越大越好
            verbose: 是否打印信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.early_stop = False
        self.best_score = None
        
    def __call__(self, metric: float):
        score = -metric if self.mode == 'min' else metric
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def save_predictions(y_true: List[int], y_pred: List[int], 
                    texts: List[str], filepath: str):
    """保存预测结果"""
    import pandas as pd
    
    df = pd.DataFrame({
        'text': texts,
        'true_label': y_true,
        'predicted_label': y_pred
    })
    
    df.to_csv(filepath, index=False, encoding='utf-8')


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    import json
    
    default_config = {
        'model': {
            'embedding_dim': 128,
            'hidden_dim': 256,
            'num_layers': 2,
            'dropout': 0.3
        },
        'training': {
            'learning_rate': 0.001,
            'batch_size': 32,
            'num_epochs': 20,
            'patience': 5
        },
        'data': {
            'max_vocab_size': 10000,
            'max_length': 128,
            'min_freq': 2
        }
    }
    
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 合并默认配置
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if sub_key not in config[key]:
                        config[key][sub_key] = sub_value
    else:
        config = default_config
    
    return config


def predict_sentiment(model: torch.nn.Module, text: str, vocab: Dict[str, int],
                     label_encoder, device: str = 'cpu', max_length: int = 128) -> str:
    """预测单个文本的情感"""
    model.eval()
    
    # 预处理文本
    tokens = tokenize_chinese(text)
    token_ids = [vocab.get(token, vocab.get('<UNK>', 1)) for token in tokens]
    
    # 截断或填充
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]
        length = max_length
    else:
        length = len(token_ids)
        token_ids.extend([0] * (max_length - len(token_ids)))
    
    # 转换为tensor
    input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
    lengths = torch.tensor([length], dtype=torch.long).to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(input_ids, lengths)
        predicted = torch.argmax(outputs, dim=1).cpu().numpy()
    
    # 转换为标签
    predicted_label = label_encoder.inverse_transform(predicted)[0]
    
    return predicted_label


def analyze_text_length_distribution(texts: List[str], save_path: str = None):
    """分析文本长度分布"""
    lengths = [len(tokenize_chinese(text)) for text in texts]
    
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Text Length (tokens)')
    plt.ylabel('Frequency')
    plt.title('Text Length Distribution')
    plt.axvline(np.mean(lengths), color='red', linestyle='--', label=f'Mean: {np.mean(lengths):.1f}')
    plt.axvline(np.median(lengths), color='green', linestyle='--', label=f'Median: {np.median(lengths):.1f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Text Length Statistics:")
    print(f"Mean: {np.mean(lengths):.2f}")
    print(f"Median: {np.median(lengths):.2f}")
    print(f"Min: {min(lengths)}")
    print(f"Max: {max(lengths)}")
    print(f"95th percentile: {np.percentile(lengths, 95):.2f}")


def create_sample_data(save_path: str = 'dataset/sample_data.csv'):
    """创建示例数据集"""
    import pandas as pd
    
    positive_texts = [
        "这部电影真的很好看，演员演技很棒！",
        "今天天气真好，心情很愉快。",
        "这个产品质量很不错，非常满意。",
        "服务态度很好，值得推荐。",
        "学习新知识让我感到很充实。",
        "朋友们都很支持我的决定。",
        "这家餐厅的菜品很美味。",
        "工作进展顺利，同事很友善。",
        "旅行很愉快，风景很美。",
        "这本书内容很有趣，受益匪浅。"
    ]
    
    negative_texts = [
        "这部电影太无聊了，浪费时间。",
        "今天心情很糟糕，什么都不顺。",
        "这个产品质量很差，很失望。",
        "服务态度很恶劣，不推荐。",
        "学习压力太大，感到焦虑。",
        "朋友们都不理解我。",
        "这家餐厅的菜品很难吃。",
        "工作压力大，同事关系紧张。",
        "旅行遇到很多问题，很糟心。",
        "这本书内容很无聊，看不下去。"
    ]
    
    neutral_texts = [
        "这是一部普通的电影。",
        "今天是平常的一天。",
        "这个产品质量还可以。",
        "服务态度一般。",
        "学习进度正常。",
        "朋友们有不同的看法。",
        "这家餐厅的菜品一般。",
        "工作如常进行。",
        "旅行还算顺利。",
        "这本书内容还行。"
    ]
    
    # 扩展数据
    texts = []
    labels = []
    
    for i in range(300):
        if i % 3 == 0:
            texts.append(positive_texts[i % len(positive_texts)])
            labels.append('positive')
        elif i % 3 == 1:
            texts.append(negative_texts[i % len(negative_texts)])
            labels.append('negative')
        else:
            texts.append(neutral_texts[i % len(neutral_texts)])
            labels.append('neutral')
    
    # 创建DataFrame
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False, encoding='utf-8')
    
    print(f"示例数据已保存到: {save_path}")
    print(f"数据大小: {len(df)}")
    print(f"标签分布:\n{df['label'].value_counts()}")
    
    return df