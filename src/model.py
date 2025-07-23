import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
import numpy as np
from typing import Optional, Tuple


class LightGCNConv(MessagePassing):
    """LightGCN卷积层"""
    
    def __init__(self, **kwargs):
        super().__init__(aggr='add', **kwargs)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, embedding_dim]
            edge_index: 边索引 [2, num_edges]
            edge_weight: 边权重 [num_edges] (可选)
            
        Returns:
            更新后的节点特征
        """
        # 计算归一化
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # 执行消息传递
        return self.propagate(edge_index, x=x, norm=norm, edge_weight=edge_weight)
    
    def message(self, x_j: torch.Tensor, norm: torch.Tensor, 
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """消息计算"""
        if edge_weight is not None:
            return norm.view(-1, 1) * x_j * edge_weight.view(-1, 1)
        else:
            return norm.view(-1, 1) * x_j


class LightGCN(nn.Module):
    """LightGCN模型"""
    
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64, 
                 n_layers: int = 3, dropout: float = 0.1):
        """
        初始化LightGCN模型
        
        Args:
            n_users: 用户数量
            n_items: 物品数量 
            embedding_dim: 嵌入维度
            n_layers: GCN层数
            dropout: dropout率
        """
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.dropout = dropout
        
        # 用户和物品嵌入
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # LightGCN卷积层
        self.gcn_layers = nn.ModuleList([
            LightGCNConv() for _ in range(n_layers)
        ])
        
        # dropout层
        self.dropout_layer = nn.Dropout(dropout)
        
        # 初始化参数
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型参数"""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
    def forward(self, edge_index: torch.Tensor, 
                edge_weight: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            edge_index: 边索引 [2, num_edges]
            edge_weight: 边权重 [num_edges] (可选)
            
        Returns:
            user_embeddings: 用户最终嵌入 [n_users, embedding_dim]
            item_embeddings: 物品最终嵌入 [n_items, embedding_dim] 
        """
        # 获取初始嵌入
        user_emb = self.user_embedding.weight  # [n_users, embedding_dim]
        item_emb = self.item_embedding.weight  # [n_items, embedding_dim]
        
        # 合并用户和物品嵌入
        all_emb = torch.cat([user_emb, item_emb], dim=0)  # [n_users + n_items, embedding_dim]
        
        # 存储每层的嵌入
        emb_layers = [all_emb]
        
        # 多层图卷积
        for layer in self.gcn_layers:
            all_emb = layer(all_emb, edge_index, edge_weight)
            all_emb = self.dropout_layer(all_emb)
            emb_layers.append(all_emb)
        
        # 层聚合：对所有层的嵌入取平均
        final_emb = torch.stack(emb_layers, dim=0).mean(dim=0)
        
        # 分离用户和物品嵌入
        user_embeddings = final_emb[:self.n_users]
        item_embeddings = final_emb[self.n_users:]
        
        return user_embeddings, item_embeddings
    
    def predict(self, user_indices: torch.Tensor, item_indices: torch.Tensor,
                user_embeddings: torch.Tensor, item_embeddings: torch.Tensor) -> torch.Tensor:
        """
        预测用户对物品的偏好分数
        
        Args:
            user_indices: 用户索引 [batch_size]
            item_indices: 物品索引 [batch_size]
            user_embeddings: 用户嵌入 [n_users, embedding_dim]
            item_embeddings: 物品嵌入 [n_items, embedding_dim]
            
        Returns:
            预测分数 [batch_size]
        """
        user_emb = user_embeddings[user_indices]  # [batch_size, embedding_dim]
        item_emb = item_embeddings[item_indices]  # [batch_size, embedding_dim]
        
        # 计算内积
        scores = torch.sum(user_emb * item_emb, dim=1)  # [batch_size]
        
        return scores
    
    def recommend(self, user_idx: int, user_embeddings: torch.Tensor, 
                  item_embeddings: torch.Tensor, k: int = 20) -> torch.Tensor:
        """
        为指定用户生成top-k推荐
        
        Args:
            user_idx: 用户索引
            user_embeddings: 用户嵌入
            item_embeddings: 物品嵌入
            k: 推荐数量
            
        Returns:
            推荐物品索引 [k]
        """
        user_emb = user_embeddings[user_idx]  # [embedding_dim]
        
        # 计算与所有物品的相似度
        scores = torch.matmul(user_emb, item_embeddings.t())  # [n_items]
        
        # 获取top-k物品
        _, top_items = torch.topk(scores, k)
        
        return top_items
    
    def get_all_ratings(self, user_embeddings: torch.Tensor, 
                       item_embeddings: torch.Tensor) -> torch.Tensor:
        """
        获取所有用户对所有物品的评分矩阵
        
        Args:
            user_embeddings: 用户嵌入 [n_users, embedding_dim]
            item_embeddings: 物品嵌入 [n_items, embedding_dim]
            
        Returns:
            评分矩阵 [n_users, n_items]
        """
        return torch.matmul(user_embeddings, item_embeddings.t())


