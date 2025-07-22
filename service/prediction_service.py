import os
import sys
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import json
import pickle
from flask import Flask, request, jsonify
import logging

# 添加项目根目录到路径
sys.path.append('..')

from src.model import LightGCN
from src.utils import setup_logger


class LightGCNRecommendationService:
    """LightGCN推荐服务"""
    
    def __init__(self, model_path: str = '../outputs/models/best_model.pt'):
        """
        初始化推荐服务
        
        Args:
            model_path: 模型文件路径
        """
        self.model_path = model_path
        self.logger = setup_logger('RecommendService')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 模型相关
        self.model = None
        self.user_embeddings = None
        self.item_embeddings = None
        
        # 数据映射
        self.user_mapping = None
        self.item_mapping = None
        self.reverse_user_mapping = None
        self.reverse_item_mapping = None
        self.n_users = 0
        self.n_items = 0
        
        # 加载模型
        self.load_model()
        
    def load_model(self):
        """加载训练好的模型"""
        try:
            self.logger.info(f"加载模型: {self.model_path}")
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
            # 加载checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 提取配置信息
            config = checkpoint['config']
            self.n_users = checkpoint['n_users']
            self.n_items = checkpoint['n_items']
            self.user_mapping = checkpoint['user_mapping']
            self.item_mapping = checkpoint['item_mapping']
            
            # 创建反向映射
            self.reverse_user_mapping = {v: k for k, v in self.user_mapping.items()}
            self.reverse_item_mapping = {v: k for k, v in self.item_mapping.items()}
            
            # 初始化模型
            model_config = config['model']
            self.model = LightGCN(
                n_users=self.n_users,
                n_items=self.n_items,
                embedding_dim=model_config['embedding_dim'],
                n_layers=model_config['n_layers'],
                dropout=model_config['dropout']
            )
            
            # 加载模型权重
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # 预计算用户和物品嵌入
            self._precompute_embeddings()
            
            self.logger.info("模型加载成功")
            self.logger.info(f"用户数: {self.n_users}, 物品数: {self.n_items}")
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            raise
    
    def _precompute_embeddings(self):
        """预计算用户和物品嵌入"""
        try:
            # 加载边索引
            edge_index_path = '../dataset/processed/edge_index.pt'
            if os.path.exists(edge_index_path):
                edge_index = torch.load(edge_index_path, map_location=self.device)
            else:
                # 如果没有边索引，创建一个空的边索引
                edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                self.logger.warning("未找到边索引文件，使用空边索引")
            
            with torch.no_grad():
                self.user_embeddings, self.item_embeddings = self.model(edge_index)
                
            self.logger.info("用户和物品嵌入预计算完成")
            
        except Exception as e:
            self.logger.error(f"嵌入预计算失败: {e}")
            # 使用初始嵌入作为后备
            with torch.no_grad():
                self.user_embeddings = self.model.user_embedding.weight
                self.item_embeddings = self.model.item_embedding.weight
    
    def get_user_recommendations(self, user_id: int, k: int = 20, 
                               exclude_interacted: bool = True) -> List[Dict[str, Any]]:
        """
        为指定用户生成推荐
        
        Args:
            user_id: 用户ID
            k: 推荐数量
            exclude_interacted: 是否排除已交互的物品
            
        Returns:
            推荐结果列表
        """
        try:
            # 检查用户是否存在
            if user_id not in self.user_mapping:
                self.logger.warning(f"用户 {user_id} 不存在于训练数据中")
                return self._get_popular_items(k)
            
            user_idx = self.user_mapping[user_id]
            
            # 获取用户嵌入
            user_emb = self.user_embeddings[user_idx]
            
            # 计算与所有物品的相似度
            scores = torch.matmul(user_emb, self.item_embeddings.t())
            
            # 如果需要排除已交互的物品
            if exclude_interacted:
                # 这里简化处理，实际应用中需要加载用户的历史交互
                pass
            
            # 获取top-k物品
            _, top_item_indices = torch.topk(scores, k)
            
            # 转换为推荐结果
            recommendations = []
            for rank, item_idx in enumerate(top_item_indices.cpu().numpy()):
                original_item_id = self.reverse_item_mapping[item_idx]
                score = scores[item_idx].item()
                
                recommendations.append({
                    'item_id': original_item_id,
                    'score': float(score),
                    'rank': rank + 1
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"为用户 {user_id} 生成推荐失败: {e}")
            return []
    
    def get_batch_recommendations(self, user_ids: List[int], k: int = 20) -> Dict[int, List[Dict[str, Any]]]:
        """
        批量生成推荐
        
        Args:
            user_ids: 用户ID列表
            k: 每个用户的推荐数量
            
        Returns:
            用户推荐结果字典
        """
        results = {}
        
        for user_id in user_ids:
            results[user_id] = self.get_user_recommendations(user_id, k)
        
        return results
    
    def get_item_similarities(self, item_id: int, k: int = 20) -> List[Dict[str, Any]]:
        """
        获取与指定物品相似的物品
        
        Args:
            item_id: 物品ID
            k: 相似物品数量
            
        Returns:
            相似物品列表
        """
        try:
            if item_id not in self.item_mapping:
                self.logger.warning(f"物品 {item_id} 不存在于训练数据中")
                return []
            
            item_idx = self.item_mapping[item_id]
            
            # 获取物品嵌入
            item_emb = self.item_embeddings[item_idx]
            
            # 计算与所有物品的相似度
            similarities = torch.matmul(item_emb, self.item_embeddings.t())
            
            # 获取top-k相似物品（排除自己）
            _, top_item_indices = torch.topk(similarities, k + 1)
            top_item_indices = top_item_indices[1:]  # 排除自己
            
            # 转换为结果
            similar_items = []
            for rank, similar_item_idx in enumerate(top_item_indices.cpu().numpy()):
                original_item_id = self.reverse_item_mapping[similar_item_idx]
                similarity = similarities[similar_item_idx].item()
                
                similar_items.append({
                    'item_id': original_item_id,
                    'similarity': float(similarity),
                    'rank': rank + 1
                })
            
            return similar_items
            
        except Exception as e:
            self.logger.error(f"获取物品 {item_id} 的相似物品失败: {e}")
            return []
    
    def _get_popular_items(self, k: int = 20) -> List[Dict[str, Any]]:
        """
        获取热门物品（冷启动用户的后备推荐）
        
        Args:
            k: 推荐数量
            
        Returns:
            热门物品列表
        """
        # 简化实现：随机选择物品
        # 实际应用中应该基于物品的受欢迎程度
        popular_items = []
        
        # 随机选择k个物品
        random_indices = np.random.choice(self.n_items, min(k, self.n_items), replace=False)
        
        for rank, item_idx in enumerate(random_indices):
            original_item_id = self.reverse_item_mapping[item_idx]
            
            popular_items.append({
                'item_id': original_item_id,
                'score': 1.0 - rank * 0.01,  # 模拟评分
                'rank': rank + 1,
                'reason': 'popular'
            })
        
        return popular_items
    
    def predict_user_item_score(self, user_id: int, item_id: int) -> float:
        """
        预测用户对物品的偏好分数
        
        Args:
            user_id: 用户ID
            item_id: 物品ID
            
        Returns:
            偏好分数
        """
        try:
            if user_id not in self.user_mapping or item_id not in self.item_mapping:
                return 0.0
            
            user_idx = self.user_mapping[user_id]
            item_idx = self.item_mapping[item_id]
            
            user_emb = self.user_embeddings[user_idx]
            item_emb = self.item_embeddings[item_idx]
            
            score = torch.dot(user_emb, item_emb).item()
            return float(score)
            
        except Exception as e:
            self.logger.error(f"预测用户 {user_id} 对物品 {item_id} 的评分失败: {e}")
            return 0.0
    
    def get_model_stats(self) -> Dict[str, Any]:
        """获取模型统计信息"""
        return {
            'n_users': self.n_users,
            'n_items': self.n_items,
            'embedding_dim': self.model.embedding_dim,
            'n_layers': self.model.n_layers,
            'device': str(self.device),
            'model_loaded': self.model is not None
        }


# Flask Web API
app = Flask(__name__)

# 全局推荐服务实例
recommendation_service = None


@app.before_first_request
def initialize_service():
    """初始化推荐服务"""
    global recommendation_service
    try:
        recommendation_service = LightGCNRecommendationService()
        app.logger.info("推荐服务初始化成功")
    except Exception as e:
        app.logger.error(f"推荐服务初始化失败: {e}")


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'service': 'LightGCN Recommendation Service',
        'model_loaded': recommendation_service is not None
    })


@app.route('/recommend/<int:user_id>', methods=['GET'])
def recommend_for_user(user_id):
    """为用户生成推荐"""
    try:
        k = request.args.get('k', default=20, type=int)
        
        if recommendation_service is None:
            return jsonify({'error': 'Service not initialized'}), 500
        
        recommendations = recommendation_service.get_user_recommendations(user_id, k)
        
        return jsonify({
            'user_id': user_id,
            'recommendations': recommendations,
            'count': len(recommendations)
        })
        
    except Exception as e:
        app.logger.error(f"推荐生成失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/recommend/batch', methods=['POST'])
def batch_recommend():
    """批量推荐"""
    try:
        data = request.get_json()
        user_ids = data.get('user_ids', [])
        k = data.get('k', 20)
        
        if not user_ids:
            return jsonify({'error': 'user_ids is required'}), 400
        
        if recommendation_service is None:
            return jsonify({'error': 'Service not initialized'}), 500
        
        results = recommendation_service.get_batch_recommendations(user_ids, k)
        
        return jsonify({
            'results': results,
            'user_count': len(user_ids)
        })
        
    except Exception as e:
        app.logger.error(f"批量推荐失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/similar/<int:item_id>', methods=['GET'])
def get_similar_items(item_id):
    """获取相似物品"""
    try:
        k = request.args.get('k', default=20, type=int)
        
        if recommendation_service is None:
            return jsonify({'error': 'Service not initialized'}), 500
        
        similar_items = recommendation_service.get_item_similarities(item_id, k)
        
        return jsonify({
            'item_id': item_id,
            'similar_items': similar_items,
            'count': len(similar_items)
        })
        
    except Exception as e:
        app.logger.error(f"相似物品查询失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict_score():
    """预测用户对物品的评分"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        item_id = data.get('item_id')
        
        if user_id is None or item_id is None:
            return jsonify({'error': 'user_id and item_id are required'}), 400
        
        if recommendation_service is None:
            return jsonify({'error': 'Service not initialized'}), 500
        
        score = recommendation_service.predict_user_item_score(user_id, item_id)
        
        return jsonify({
            'user_id': user_id,
            'item_id': item_id,
            'predicted_score': score
        })
        
    except Exception as e:
        app.logger.error(f"评分预测失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    """获取模型统计信息"""
    try:
        if recommendation_service is None:
            return jsonify({'error': 'Service not initialized'}), 500
        
        stats = recommendation_service.get_model_stats()
        return jsonify(stats)
        
    except Exception as e:
        app.logger.error(f"获取统计信息失败: {e}")
        return jsonify({'error': str(e)}), 500


def main():
    """启动推荐服务"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LightGCN推荐服务')
    parser.add_argument('--host', default='0.0.0.0', help='服务主机地址')
    parser.add_argument('--port', type=int, default=5000, help='服务端口')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    parser.add_argument('--model-path', default='../outputs/models/best_model.pt', 
                       help='模型文件路径')
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    print(f"启动LightGCN推荐服务...")
    print(f"模型路径: {args.model_path}")
    print(f"服务地址: http://{args.host}:{args.port}")
    
    # 初始化全局服务
    global recommendation_service
    try:
        recommendation_service = LightGCNRecommendationService(args.model_path)
        print("推荐服务初始化成功!")
    except Exception as e:
        print(f"推荐服务初始化失败: {e}")
        return
    
    # 启动Flask应用
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()