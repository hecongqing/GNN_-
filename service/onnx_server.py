#!/usr/bin/env python3
"""
基于ONNX的LightGCN推荐服务器
提供HTTP API接口进行推理
"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Any, Optional
import argparse
import logging
from pathlib import Path

from flask import Flask, request, jsonify, abort
import onnxruntime as ort

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils import setup_logger


class ONNXRecommendationServer:
    """基于ONNX的推荐服务器"""
    
    def __init__(self, model_path: str, metadata_path: str = None):
        """
        初始化ONNX推荐服务器
        
        Args:
            model_path: ONNX模型文件路径
            metadata_path: 模型元数据文件路径
        """
        self.model_path = model_path
        self.metadata_path = metadata_path or model_path.replace('.onnx', '_metadata.json')
        self.logger = setup_logger('ONNXServer')
        
        # 模型相关
        self.session = None
        self.input_name = None
        self.output_names = None
        
        # 元数据
        self.n_users = 0
        self.n_items = 0
        self.embedding_dim = 64
        self.n_layers = 3
        
        # 用户-物品交互数据（用于推理）
        self.edge_index = None
        
        # 初始化
        self.load_model()
        self.load_metadata()
        self.prepare_inference_data()
        
    def load_model(self):
        """加载ONNX模型"""
        self.logger.info(f"加载ONNX模型: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ONNX模型文件不存在: {self.model_path}")
        
        try:
            # 创建ONNX Runtime会话
            self.session = ort.InferenceSession(self.model_path)
            
            # 获取输入输出信息
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            self.logger.info(f"模型加载成功，输入: {self.input_name}, 输出: {self.output_names}")
            
        except Exception as e:
            self.logger.error(f"加载ONNX模型失败: {e}")
            raise
    
    def load_metadata(self):
        """加载模型元数据"""
        if os.path.exists(self.metadata_path):
            self.logger.info(f"加载元数据: {self.metadata_path}")
            
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            self.n_users = metadata.get('n_users', 1000)
            self.n_items = metadata.get('n_items', 500)
            self.embedding_dim = metadata.get('embedding_dim', 64)
            self.n_layers = metadata.get('n_layers', 3)
            
            self.logger.info(f"元数据加载完成 - 用户数: {self.n_users}, 物品数: {self.n_items}")
        else:
            self.logger.warning(f"元数据文件不存在: {self.metadata_path}，使用默认值")
    
    def prepare_inference_data(self):
        """准备推理数据"""
        self.logger.info("准备推理数据...")
        
        # 创建示例边索引（实际应用中应该从数据库或文件加载）
        # 这里创建一个稀疏的用户-物品交互矩阵
        num_edges = min(self.n_users * 5, 10000)
        
        user_indices = np.random.randint(0, self.n_users, num_edges)
        item_indices = np.random.randint(0, self.n_items, num_edges) + self.n_users
        
        self.edge_index = np.stack([user_indices, item_indices])
        
        self.logger.info(f"推理数据准备完成，边数: {num_edges}")
    
    def get_user_recommendations(self, user_id: int, top_k: int = 10) -> List[int]:
        """
        为指定用户生成推荐
        
        Args:
            user_id: 用户ID
            top_k: 推荐物品数量
            
        Returns:
            推荐的物品ID列表
        """
        if user_id < 0 or user_id >= self.n_users:
            raise ValueError(f"用户ID {user_id} 超出范围 [0, {self.n_users})")
        
        try:
            # 运行ONNX推理
            outputs = self.session.run(
                self.output_names,
                {self.input_name: self.edge_index}
            )
            
            user_embeddings, item_embeddings = outputs
            
            # 计算用户与所有物品的相似度分数
            user_emb = user_embeddings[user_id:user_id+1]  # [1, embedding_dim]
            scores = np.dot(user_emb, item_embeddings.T).squeeze()  # [n_items]
            
            # 获取Top-K推荐
            top_items = np.argsort(scores)[-top_k:][::-1]
            
            return top_items.tolist()
            
        except Exception as e:
            self.logger.error(f"推理失败: {e}")
            raise
    
    def get_batch_recommendations(self, user_ids: List[int], top_k: int = 10) -> Dict[int, List[int]]:
        """
        批量生成推荐
        
        Args:
            user_ids: 用户ID列表
            top_k: 推荐物品数量
            
        Returns:
            用户ID到推荐列表的映射
        """
        recommendations = {}
        
        for user_id in user_ids:
            try:
                recommendations[user_id] = self.get_user_recommendations(user_id, top_k)
            except Exception as e:
                self.logger.error(f"用户 {user_id} 推荐失败: {e}")
                recommendations[user_id] = []
        
        return recommendations
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            'status': 'healthy',
            'model_path': self.model_path,
            'n_users': self.n_users,
            'n_items': self.n_items,
            'embedding_dim': self.embedding_dim,
            'n_layers': self.n_layers
        }


# Flask应用
app = Flask(__name__)
server = None


@app.route('/health', methods=['GET'])
def health():
    """健康检查接口"""
    if server is None:
        abort(503)
    
    return jsonify(server.health_check())


@app.route('/recommend', methods=['POST'])
def recommend():
    """单用户推荐接口"""
    if server is None:
        abort(503)
    
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        top_k = data.get('top_k', 10)
        
        if user_id is None:
            return jsonify({'error': '缺少user_id参数'}), 400
        
        recommendations = server.get_user_recommendations(user_id, top_k)
        
        return jsonify({
            'user_id': user_id,
            'recommendations': recommendations,
            'top_k': top_k
        })
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        app.logger.error(f"推荐失败: {e}")
        return jsonify({'error': '推荐服务异常'}), 500


@app.route('/recommend/batch', methods=['POST'])
def recommend_batch():
    """批量推荐接口"""
    if server is None:
        abort(503)
    
    try:
        data = request.get_json()
        user_ids = data.get('user_ids', [])
        top_k = data.get('top_k', 10)
        
        if not user_ids:
            return jsonify({'error': '缺少user_ids参数'}), 400
        
        if len(user_ids) > 100:  # 限制批量大小
            return jsonify({'error': '批量请求不能超过100个用户'}), 400
        
        recommendations = server.get_batch_recommendations(user_ids, top_k)
        
        return jsonify({
            'recommendations': recommendations,
            'top_k': top_k
        })
        
    except Exception as e:
        app.logger.error(f"批量推荐失败: {e}")
        return jsonify({'error': '批量推荐服务异常'}), 500


@app.route('/info', methods=['GET'])
def info():
    """模型信息接口"""
    if server is None:
        abort(503)
    
    return jsonify({
        'model_type': 'LightGCN',
        'format': 'ONNX',
        'n_users': server.n_users,
        'n_items': server.n_items,
        'embedding_dim': server.embedding_dim,
        'n_layers': server.n_layers
    })


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='启动ONNX推荐服务器')
    parser.add_argument('--model-path', type=str, required=True,
                       help='ONNX模型文件路径')
    parser.add_argument('--metadata-path', type=str,
                       help='模型元数据文件路径')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='服务器主机地址')
    parser.add_argument('--port', type=int, default=8080,
                       help='服务器端口')
    parser.add_argument('--debug', action='store_true',
                       help='启用调试模式')
    
    args = parser.parse_args()
    
    try:
        # 初始化服务器
        global server
        server = ONNXRecommendationServer(args.model_path, args.metadata_path)
        
        # 配置日志
        if not args.debug:
            logging.getLogger('werkzeug').setLevel(logging.WARNING)
        
        print(f"🚀 ONNX推荐服务器启动成功！")
        print(f"📍 地址: http://{args.host}:{args.port}")
        print(f"📋 健康检查: http://{args.host}:{args.port}/health")
        print(f"🔍 模型信息: http://{args.host}:{args.port}/info")
        
        # 启动Flask应用
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True
        )
        
    except Exception as e:
        print(f"❌ 服务器启动失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())