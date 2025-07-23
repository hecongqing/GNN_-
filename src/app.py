#!/usr/bin/env python3
"""
LightGCN推荐系统主应用程序
提供训练和推理功能
"""

import os
import sys
import argparse
import json
import torch
from pathlib import Path

from .train import LightGCNTrainingPipeline
from .model import LightGCN
from .dataset import AliRecommendDataset
from .utils import setup_logger


def train_model(config_path: str = "config.json"):
    """训练模型"""
    logger = setup_logger('App')
    
    logger.info("开始训练LightGCN模型...")
    
    # 创建训练管道
    pipeline = LightGCNTrainingPipeline()
    
    # 加载配置
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            logger.info(f"加载配置文件: {config_path}")
    else:
        logger.warning("配置文件不存在，使用默认配置")
        config = {}
    
    # 执行训练
    pipeline.train(config)
    logger.info("训练完成！")


def inference(model_path: str, user_id: int, top_k: int = 10):
    """推理/预测"""
    logger = setup_logger('App')
    
    logger.info(f"开始推理，用户ID: {user_id}, Top-K: {top_k}")
    
    # 加载数据集
    dataset = AliRecommendDataset()
    dataset.load_data()
    dataset.preprocess_data()
    
    # 加载模型
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return None
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model = LightGCN(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        embedding_dim=checkpoint.get('embedding_dim', 64),
        n_layers=checkpoint.get('n_layers', 3)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 生成推荐
    with torch.no_grad():
        user_emb, item_emb = model(dataset.edge_index)
        scores = torch.mm(user_emb[user_id:user_id+1], item_emb.t())
        _, top_items = torch.topk(scores, top_k)
        
    recommendations = top_items.squeeze().tolist()
    logger.info(f"推荐结果: {recommendations}")
    
    return recommendations


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LightGCN推荐系统')
    parser.add_argument('--mode', choices=['train', 'inference'], required=True,
                       help='运行模式：train(训练) 或 inference(推理)')
    parser.add_argument('--config', type=str, default='config.json',
                       help='配置文件路径')
    parser.add_argument('--model-path', type=str, default='outputs/models/best_model.pt',
                       help='模型文件路径')
    parser.add_argument('--user-id', type=int, default=0,
                       help='推理时的用户ID')
    parser.add_argument('--top-k', type=int, default=10,
                       help='推荐商品数量')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/logs', exist_ok=True)
    
    if args.mode == 'train':
        train_model(args.config)
    elif args.mode == 'inference':
        recommendations = inference(args.model_path, args.user_id, args.top_k)
        if recommendations:
            print(f"用户 {args.user_id} 的推荐结果: {recommendations}")


if __name__ == "__main__":
    main()