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


class BPRLoss(nn.Module):
    """贝叶斯个性化排序损失函数"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        """
        计算BPR损失
        
        Args:
            pos_scores: 正样本分数 [batch_size]
            neg_scores: 负样本分数 [batch_size]
            
        Returns:
            BPR损失
        """
        diff = pos_scores - neg_scores
        loss = -torch.log(torch.sigmoid(diff)).mean()
        return loss


class LightGCNTrainer:
    """LightGCN训练器"""
    
    def __init__(self, model: LightGCN, device: torch.device = None):
        """
        初始化训练器
        
        Args:
            model: LightGCN模型
            device: 计算设备
        """
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 损失函数
        self.bpr_loss = BPRLoss()
        
    def create_bpr_batch(self, interaction_matrix: np.ndarray, 
                        batch_size: int = 1024) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        创建BPR训练批次
        
        Args:
            interaction_matrix: 交互矩阵
            batch_size: 批次大小
            
        Returns:
            user_indices: 用户索引
            pos_item_indices: 正样本物品索引
            neg_item_indices: 负样本物品索引
        """
        # 获取所有正样本
        users, pos_items = np.where(interaction_matrix > 0)
        n_positive = len(users)
        
        # 随机采样批次
        batch_indices = np.random.choice(n_positive, min(batch_size, n_positive), replace=False)
        batch_users = users[batch_indices]
        batch_pos_items = pos_items[batch_indices]
        
        # 为每个正样本生成负样本
        batch_neg_items = []
        for user_idx in batch_users:
            neg_item = np.random.randint(0, self.model.n_items)
            while interaction_matrix[user_idx, neg_item] > 0:
                neg_item = np.random.randint(0, self.model.n_items)
            batch_neg_items.append(neg_item)
        
        return (
            torch.tensor(batch_users, dtype=torch.long, device=self.device),
            torch.tensor(batch_pos_items, dtype=torch.long, device=self.device),
            torch.tensor(batch_neg_items, dtype=torch.long, device=self.device)
        )
    
    def train_epoch(self, edge_index: torch.Tensor, interaction_matrix: np.ndarray,
                   optimizer: torch.optim.Optimizer, batch_size: int = 1024,
                   n_batches: int = 50) -> float:
        """
        训练一个epoch
        
        Args:
            edge_index: 边索引
            interaction_matrix: 交互矩阵
            optimizer: 优化器
            batch_size: 批次大小
            n_batches: 批次数量
            
        Returns:
            平均损失
        """
        self.model.train()
        total_loss = 0.0
        
        for _ in range(n_batches):
            optimizer.zero_grad()
            
            # 前向传播获取嵌入
            user_embeddings, item_embeddings = self.model(edge_index)
            
            # 创建训练批次
            user_indices, pos_item_indices, neg_item_indices = self.create_bpr_batch(
                interaction_matrix, batch_size
            )
            
            # 计算预测分数
            pos_scores = self.model.predict(user_indices, pos_item_indices, 
                                          user_embeddings, item_embeddings)
            neg_scores = self.model.predict(user_indices, neg_item_indices,
                                          user_embeddings, item_embeddings)
            
            # 计算损失
            loss = self.bpr_loss(pos_scores, neg_scores)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / n_batches