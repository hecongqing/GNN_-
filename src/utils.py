import torch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import Dict, List, Tuple, Any
import logging
import os
from datetime import datetime


def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """设置日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除现有handlers
    logger.handlers.clear()
    
    # 创建formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def create_user_item_mapping(df: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int], int, int]:
    """
    创建用户和物品的ID映射
    
    Args:
        df: 包含user_id和item_id的DataFrame
        
    Returns:
        user_mapping: 原始user_id到连续索引的映射
        item_mapping: 原始item_id到连续索引的映射
        n_users: 用户数量
        n_items: 物品数量
    """
    unique_users = df['user_id'].unique()
    unique_items = df['item_id'].unique()
    
    user_mapping = {uid: idx for idx, uid in enumerate(unique_users)}
    item_mapping = {iid: idx for idx, iid in enumerate(unique_items)}
    
    return user_mapping, item_mapping, len(unique_users), len(unique_items)


def create_interaction_matrix(df: pd.DataFrame, user_mapping: Dict[int, int], 
                            item_mapping: Dict[int, int], n_users: int, n_items: int) -> np.ndarray:
    """
    创建用户-物品交互矩阵
    
    Args:
        df: 包含用户物品交互的DataFrame
        user_mapping: 用户ID映射
        item_mapping: 物品ID映射
        n_users: 用户数量
        n_items: 物品数量
        
    Returns:
        交互矩阵 (n_users, n_items)
    """
    interaction_matrix = np.zeros((n_users, n_items))
    
    for _, row in df.iterrows():
        user_idx = user_mapping[row['user_id']]
        item_idx = item_mapping[row['item_id']]
        interaction_matrix[user_idx, item_idx] = 1
        
    return interaction_matrix


def build_graph_edges(interaction_matrix: np.ndarray) -> torch.Tensor:
    """
    构建二部图的边索引
    
    Args:
        interaction_matrix: 用户-物品交互矩阵
        
    Returns:
        edge_index: 边索引 [2, num_edges]
    """
    n_users, n_items = interaction_matrix.shape
    
    # 找到所有非零交互
    user_indices, item_indices = np.nonzero(interaction_matrix)
    
    # 物品节点的索引需要偏移用户数量
    item_indices_shifted = item_indices + n_users
    
    # 构建无向图：用户->物品 和 物品->用户
    edge_index = np.vstack([
        np.concatenate([user_indices, item_indices_shifted]),
        np.concatenate([item_indices_shifted, user_indices])
    ])
    
    return torch.tensor(edge_index, dtype=torch.long)


def compute_metrics(predictions: np.ndarray, ground_truth: np.ndarray, k: int = 20) -> Dict[str, float]:
    """
    计算推荐系统评价指标
    
    Args:
        predictions: 预测结果 [n_users, n_items]
        ground_truth: 真实标签 [n_users, n_items]
        k: top-k推荐
        
    Returns:
        包含precision, recall, f1的字典
    """
    n_users = predictions.shape[0]
    
    precisions = []
    recalls = []
    
    for user_idx in range(n_users):
        # 获取top-k推荐
        user_pred = predictions[user_idx]
        user_true = ground_truth[user_idx]
        
        # 如果用户没有真实交互，跳过
        if user_true.sum() == 0:
            continue
            
        # 获取top-k物品索引
        top_k_items = np.argsort(user_pred)[-k:]
        
        # 计算命中数
        hits = sum(user_true[item] for item in top_k_items)
        
        # 计算precision和recall
        precision = hits / k if k > 0 else 0
        recall = hits / user_true.sum() if user_true.sum() > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0
    avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    
    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1
    }


def save_predictions(predictions: List[Tuple[int, int]], output_path: str):
    """
    保存预测结果
    
    Args:
        predictions: [(user_id, item_id), ...] 的预测结果列表
        output_path: 输出文件路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("user_id\titem_id\n")
        for user_id, item_id in predictions:
            f.write(f"{user_id}\t{item_id}\n")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    import json
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # 默认配置
        return {
            "model": {
                "embedding_dim": 64,
                "n_layers": 3,
                "dropout": 0.1
            },
            "training": {
                "batch_size": 1024,
                "learning_rate": 0.001,
                "epochs": 100,
                "weight_decay": 1e-4
            },
            "evaluation": {
                "k": 20,
                "test_ratio": 0.2
            }
        }


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def set_random_seed(seed: int = 42):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False