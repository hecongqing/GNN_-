"""
LightGCN训练示例

这个示例展示了如何使用分离后的训练模块来训练LightGCN模型
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix

# 导入模型和训练器
from src.model import LightGCN
from src.train import LightGCNTrainer, BPRLoss


def create_sample_data(n_users=100, n_items=200, n_interactions=1000):
    """创建示例数据"""
    # 随机生成用户-物品交互
    user_ids = np.random.randint(0, n_users, n_interactions)
    item_ids = np.random.randint(0, n_items, n_interactions)
    
    # 创建交互矩阵
    interaction_matrix = np.zeros((n_users, n_items))
    for u, i in zip(user_ids, item_ids):
        interaction_matrix[u, i] = 1.0
    
    # 创建二分图的边索引
    users, items = np.where(interaction_matrix > 0)
    
    # 构建边：用户节点 -> 物品节点 (物品节点索引需要偏移n_users)
    edges = []
    for u, i in zip(users, items):
        edges.append([u, n_users + i])  # 用户到物品
        edges.append([n_users + i, u])  # 物品到用户
    
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    
    return interaction_matrix, edge_index


def main():
    """主训练流程"""
    print("=" * 60)
    print("LightGCN 训练示例")
    print("=" * 60)
    
    # 设置参数
    n_users = 100
    n_items = 200
    embedding_dim = 64
    n_layers = 3
    
    # 创建示例数据
    print("1. 创建示例数据...")
    interaction_matrix, edge_index = create_sample_data(n_users, n_items)
    print(f"   用户数: {n_users}")
    print(f"   物品数: {n_items}")
    print(f"   交互数: {interaction_matrix.sum():.0f}")
    
    # 创建模型
    print("\n2. 创建LightGCN模型...")
    model = LightGCN(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=embedding_dim,
        n_layers=n_layers,
        dropout=0.1
    )
    print(f"   嵌入维度: {embedding_dim}")
    print(f"   GCN层数: {n_layers}")
    print(f"   模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器
    print("\n3. 创建训练器...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = LightGCNTrainer(model, device)
    print(f"   设备: {device}")
    
    # 开始训练
    print("\n4. 开始训练...")
    history = trainer.train(
        edge_index=edge_index,
        interaction_matrix=interaction_matrix,
        n_epochs=20,
        lr=0.001,
        weight_decay=1e-4,
        batch_size=512,
        n_batches=10,
        verbose=True,
        eval_every=5
    )
    
    # 训练完成后进行推荐
    print("\n5. 生成推荐...")
    model.eval()
    with torch.no_grad():
        user_embeddings, item_embeddings = model(edge_index.to(device))
        
        # 为用户0生成推荐
        user_id = 0
        top_items = model.recommend(
            user_id, user_embeddings, item_embeddings, k=10
        ).cpu().numpy()
        
        print(f"   为用户 {user_id} 推荐的top-10物品: {top_items}")
    
    # 展示训练历史
    print("\n6. 训练历史:")
    final_loss = history['loss'][-1]
    print(f"   最终损失: {final_loss:.4f}")
    print(f"   训练轮数: {len(history['loss'])}")
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()